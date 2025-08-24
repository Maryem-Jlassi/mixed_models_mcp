import os
import tempfile
import asyncio
import json
import uuid
import re
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from simple import SimplifiedCVJobHost
from groq import GroqCVJobHost
from auth import router as auth_router, get_current_user, get_current_user_optional
from db import init_mongo, close_mongo, get_db
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId

app = FastAPI(title="Simple CV/Job Orchestrator", version="0.1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

host: Optional[SimplifiedCVJobHost] = None
groq_host: Optional[GroqCVJobHost] = None


@app.on_event("startup")
async def on_startup():
    global host, groq_host
    # Init Mongo
    try:
        await init_mongo()
    except Exception:
        pass
    # Init Ollama host
    host = SimplifiedCVJobHost()
    await host.initialize()
    # Groq host will be initialized lazily on first request to /groq/workflow


@app.on_event("shutdown")
async def on_shutdown():
    global host, groq_host
    if host:
        await host.shutdown()
        host = None
    if groq_host:
        await groq_host.shutdown()
        groq_host = None
    try:
        await close_mongo()
    except Exception:
        pass

# Auth routes
app.include_router(auth_router)


@app.get("/simple/health")
async def health():
    return {"status": "ok", "components": {"fastapi": "healthy", "host_initialized": bool(host)}}


@app.get("/groq/health")
async def groq_health():
    return {"status": "ok", "components": {"fastapi": "healthy", "groq_host_initialized": bool(groq_host)}}


@app.get("/")
async def root():
    return {
        "app": "Simple CV/Job Orchestrator",
        "version": "0.1.0",
        "endpoints": {
            "health": "/simple/health",
            "workflow": "/workflow",
            "workflow_alt": "/simple/workflow",
            "groq_health": "/groq/health",
            "groq_workflow": "/groq/workflow",
        },
    }


async def _workflow_common(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    job_url: str = Form(""),
    cv_file: UploadFile = File(None),
):
    if not host:
        return JSONResponse(status_code=503, content={"success": False, "error": "host_unavailable"})

    # Build context
    ctx: Dict[str, Any] = {}
    if job_url:
        ctx["job_url"] = job_url
    elif message:
        # Auto-detect a URL inside the message if not explicitly provided
        m = re.search(r"https?://\S+", message)
        if m:
            detected = m.group(0).rstrip(').,;!"\'')
            ctx["job_url"] = detected

    temp_path = None
    if cv_file and cv_file.filename:
        try:
            contents = await cv_file.read()
            suffix = os.path.splitext(cv_file.filename)[1] or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(contents)
            tmp.flush()
            tmp.close()
            temp_path = tmp.name
            ctx["file_path"] = temp_path
            # cleanup after response finishes
            background_tasks.add_task(lambda p: (os.path.exists(p) and os.remove(p)), temp_path)
        except Exception:
            temp_path = None

    user_query = message or ""
    if not user_query and not temp_path and not job_url:
        return JSONResponse(status_code=400, content={"success": False, "error": "empty_request"})

    async def stream() -> AsyncGenerator[str, None]:
        try:
            async for line in host.process_query_stream(user_query, context=ctx):
                yield line
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
        background=background_tasks,
    )


# Generic workflow helper that targets a specific host instance
async def _workflow_for(
    target_host,
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    job_url: str = Form(""),
    cv_file: UploadFile = File(None),
    # Model is handled by the caller
    session_id: str = Form(None),
    current_user: Optional[dict] = None,
):
    if not target_host:
        return JSONResponse(status_code=503, content={"success": False, "error": "host_unavailable"})

    # Sessions in Mongo (optional)
    try:
        db = get_db()
    except Exception:
        db = None
    if not session_id:
        session_id = str(uuid.uuid4())
    owner_id = current_user.get("_id") if current_user else None
    if db is not None:
        try:
            existing = await db.sessions.find_one({"session_id": session_id})
            if not existing:
                await db.sessions.insert_one({
                    "session_id": session_id,
                    "user_id": owner_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "cv_file_id": None,
                    "cv_meta": None,
                })
            else:
                if existing.get("user_id"):
                    if not owner_id or existing.get("user_id") != owner_id:
                        raise HTTPException(status_code=403, detail="Forbidden: session not owned by user")
                elif owner_id:
                    await db.sessions.update_one({"_id": existing.get("_id")}, {"$set": {"user_id": owner_id}})
        except Exception:
            # If DB is down, skip persistence and continue
            db = None

    # Build context
    ctx: Dict[str, Any] = {}
    if job_url:
        ctx["job_url"] = job_url

    temp_path = None
    if cv_file and cv_file.filename:
        try:
            contents = await cv_file.read()
            suffix = os.path.splitext(cv_file.filename)[1] or ".pdf"
            # Upload to GridFS if DB available
            if db is not None:
                try:
                    bucket = AsyncIOMotorGridFSBucket(get_db(), bucket_name="cv_files")
                    gridfs_filename = f"{session_id}_{cv_file.filename}"
                    file_id = await bucket.upload_from_stream(gridfs_filename, contents)
                    await db.sessions.update_one(
                        {"session_id": session_id},
                        {"$set": {
                            "cv_file_id": str(file_id),
                            "cv_meta": {
                                "original_filename": cv_file.filename,
                                "content_type": cv_file.content_type,
                                "file_size": len(contents),
                            },
                            "updated_at": datetime.utcnow().isoformat(),
                        }}
                    )
                except Exception:
                    # GridFS optional
                    pass
            # Materialize temp file for tools
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(contents)
            tmp.flush()
            tmp.close()
            temp_path = tmp.name
            ctx["file_path"] = temp_path
            background_tasks.add_task(lambda p: (os.path.exists(p) and os.remove(p)), temp_path)
        except Exception:
            temp_path = None

    # Restore last CV if none provided this time
    if not temp_path:
        try:
            sess_doc = None
            if db is not None:
                sess_doc = await db.sessions.find_one({"session_id": session_id})
            existing_file_id = (sess_doc or {}).get("cv_file_id")
            if existing_file_id:
                try:
                    bucket = AsyncIOMotorGridFSBucket(get_db(), bucket_name="cv_files")
                    stream = await bucket.open_download_stream(ObjectId(existing_file_id))
                    contents = await stream.read()
                    await stream.close()
                except Exception:
                    contents = None
                suffix = ((sess_doc or {}).get("cv_meta", {}) or {}).get("original_filename", "cv.pdf")
                suffix = os.path.splitext(suffix)[1] or ".pdf"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                if contents:
                    tmp.write(contents)
                tmp.flush()
                tmp.close()
                temp_path = tmp.name
                ctx["file_path"] = temp_path
                background_tasks.add_task(lambda p: (os.path.exists(p) and os.remove(p)), temp_path)
        except Exception:
            pass

    user_query = message or ""
    if not user_query and not temp_path and not job_url:
        return JSONResponse(status_code=400, content={"success": False, "error": "empty_request"})

    async def stream() -> AsyncGenerator[str, None]:
        assistant_buffer = ""
        try:
            # Emit session id for frontend association
            yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
            async for line in target_host.process_query_stream(user_query, context=ctx):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and data.get("type") == "response":
                        # Accumulate all chunks of assistant content
                        assistant_buffer += data.get("content", "")
                except Exception:
                    pass
                yield line
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            # Persist conversation if we have an owner and an assistant response (optional)
            if assistant_buffer.strip() and owner_id and (db is not None):
                try:
                    if user_query:
                        await db.messages.insert_one({
                            "session_id": session_id,
                            "user_id": owner_id,
                            "role": "user",
                            "content": user_query,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    await db.messages.insert_one({
                        "session_id": session_id,
                        "user_id": owner_id,
                        "role": "assistant",
                        "content": assistant_buffer,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    await db.sessions.update_one({"session_id": session_id}, {"$set": {"updated_at": datetime.utcnow().isoformat()}})
                except Exception:
                    pass

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
        background=background_tasks,
    )


@app.post("/simple/workflow")
async def simple_workflow(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    job_url: str = Form(""),
    cv_file: UploadFile = File(None),
    session_id: str = Form(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    return await _workflow_for(host, background_tasks=background_tasks, message=message, job_url=job_url, cv_file=cv_file, session_id=session_id, current_user=current_user)


@app.post("/workflow")
async def workflow_alias(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    job_url: str = Form(""),
    cv_file: UploadFile = File(None),
    model: str = Form("simple"),  # 'simple' (Ollama) or 'groq'
    session_id: str = Form(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Unified workflow endpoint with model selection.

    - model="simple" or model="ollama" -> routes to SimplifiedCVJobHost (Ollama)
    - model="groq" -> routes to GroqCVJobHost (Groq API)
    """
    global groq_host
    model_norm = (model or "simple").strip().lower()
    if model_norm in ("simple", "ollama"):
        return await _workflow_for(
            host, background_tasks=background_tasks, message=message, job_url=job_url, cv_file=cv_file, session_id=session_id, current_user=current_user
        )
    elif model_norm == "groq":
        if groq_host is None:
            groq_host = GroqCVJobHost()
            await groq_host.initialize()
        return await _workflow_for(
            groq_host, background_tasks=background_tasks, message=message, job_url=job_url, cv_file=cv_file, session_id=session_id, current_user=current_user
        )
    else:
        return JSONResponse(status_code=400, content={"success": False, "error": "invalid_model"})


@app.post("/groq/workflow")
async def groq_workflow(
    background_tasks: BackgroundTasks,
    message: str = Form(""),
    job_url: str = Form(""),
    cv_file: UploadFile = File(None),
    session_id: str = Form(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    global groq_host
    if groq_host is None:
        groq_host = GroqCVJobHost()
        await groq_host.initialize()
    return await _workflow_for(groq_host, background_tasks=background_tasks, message=message, job_url=job_url, cv_file=cv_file, session_id=session_id, current_user=current_user)


@app.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    try:
        db = get_db()
    except Exception:
        return {"sessions": []}
    user_id = current_user["_id"]
    sessions: List[Dict[str, Any]] = []
    try:
        cursor = db.sessions.find({"user_id": user_id}).sort("updated_at", -1)
        async for s in cursor:
            msg_count = await db.messages.count_documents({"session_id": s.get("session_id"), "user_id": user_id})
            if msg_count == 0:
                continue
            sessions.append({
                "session_id": s.get("session_id"),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "message_count": msg_count,
                "last_message_preview": "",
            })
        # Populate preview with last user message
        for item in sessions:
            last_user_msg = await db.messages.find({"session_id": item["session_id"], "role": "user", "user_id": user_id}).sort("timestamp", -1).to_list(1)
            if last_user_msg:
                content = last_user_msg[0].get("content") or ""
                item["last_message_preview"] = (content[:120] + ("..." if len(content) > 120 else "")) if content else ""
    except Exception:
        sessions = []
    return {"sessions": sessions}


@app.get("/chat/{session_id}")
async def get_chat(session_id: str, current_user: dict = Depends(get_current_user)):
    try:
        db = get_db()
    except Exception:
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})
    try:
        sess = await db.sessions.find_one({"session_id": session_id})
        if not sess:
            return JSONResponse(status_code=404, content={"detail": "Session not found"})
        if sess.get("user_id") != current_user["_id"]:
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
        messages = await db.messages.find({"session_id": session_id, "user_id": current_user["_id"]}).sort("timestamp", 1).to_list(None)
        out_msgs = [{"role": m.get("role"), "content": m.get("content"), "timestamp": m.get("timestamp")} for m in messages]
        return {"session_id": session_id, "created_at": sess.get("created_at"), "updated_at": sess.get("updated_at"), "messages": out_msgs}
    except Exception:
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8001, reload=True)
