import os
import sys
import signal
import time
import logging
import asyncio
import uvicorn
import httpx
import uuid
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
from contextlib import asynccontextmanager
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from db import init_mongo, close_mongo, get_db
from auth import router as auth_router, get_current_user, get_current_user_optional

# Import configuration
from config import settings

# Optional LLM-first orchestrator
#from llm_orchestrator import LLMOrchestrator
# Optional local HF orchestrator (Phi-3 Mini)
try:
    from hf_host import LLMOrchestratorHF  # type: ignore
except Exception:
    LLMOrchestratorHF = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cv_processor")

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

 

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_info: Optional[Dict[str, Any]] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str
    session_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# Utility functions
# No local file cleanup needed; files are stored in Mongo GridFS
pass

 

# Global instances
llm_client = None
#llm_orchestrator: LLMOrchestrator | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    
    logger.info("🚀 Starting application...")
    # Initialize MongoDB
    await init_mongo()
    
    # UnifiedHost removed

    # Initialize LLM-first orchestrator when enabled
    global llm_orchestrator
    if getattr(settings, "LLM_ORCHESTRATION_ENABLED", False):
        try:
            provider = getattr(settings, "LLM_PROVIDER", "groq").lower()
            if provider == "hf" and LLMOrchestratorHF is not None:
                logger.info("🧠 Initializing LLM-first orchestrator (HF Transformers: Phi-3 Mini)...")
                llm_orchestrator = LLMOrchestratorHF()
            else:
                logger.info("🧠 Initializing LLM-first orchestrator (Groq)...")
                llm_orchestrator = LLMOrchestrator()
            await llm_orchestrator.initialize()
            logger.info("✅ LLM-first orchestrator initialized")
            # Pre-warm the model to avoid first-call latency/timeouts
            try:
                logger.info("🔥 Pre-warming LLM model...")
                _ = await llm_orchestrator.llm.generate(
                    prompt="ping",
                    system_prompt=None,
                    max_tokens=8,
                )
                logger.info("🔥 LLM model pre-warm complete")
            except Exception as warm_e:
                logger.warning(f"Pre-warm failed (continuing): {warm_e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM orchestrator: {e}")
            llm_orchestrator = None
    
    logger.info("✅ Application startup complete")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down application...")
    # Shutdown LLM orchestrator if used
    try:
        if llm_orchestrator:
            await llm_orchestrator.shutdown()
    finally:
        llm_orchestrator = None
    # Close MongoDB
    try:
        await close_mongo()
    except Exception:
        pass
    logger.info("✅ Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
    description="API for processing and analyzing CV/Resume documents using AI",
    lifespan=lifespan
)

# Add CORS Middleware
# Add CORS middleware with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],
)

# Auth routes
app.include_router(auth_router)

# API Endpoints

@app.post("/workflow")
@app.options("/workflow", include_in_schema=False)
async def process_workflow(
    request: Request,
    background_tasks: BackgroundTasks,
    workflow_type: str = Form(...),
    cv_file: UploadFile = File(None, alias="cv_file"),
    session_id: str = Form(None),
    message: str = Form(""),
    user_query: str = Form(""),
    cv_text: str = Form(""),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Enhanced unified workflow endpoint that handles both CV uploads and text prompts.
    Both CV upload and text prompt are optional, but at least one must be provided.
    """
    try:
        logger.info(f"Received workflow request: type={workflow_type}, session_id={session_id}")
        
        # Make CV upload optional - user can chat without uploading CV
        # Only require some form of input (message, query, or file)
        if not cv_file and not message and not user_query and not cv_text:
            raise HTTPException(
                status_code=400,
                detail="Please provide a message or upload a CV file to get started."
            )
        
        # Ensure or create session in MongoDB
        db = get_db()
        if not session_id:
            session_id = str(uuid.uuid4())
        owner_id = current_user.get("_id") if current_user else None
        existing = await db.sessions.find_one({"session_id": session_id})
        if not existing:
            doc = {
                "session_id": session_id,
                "user_id": owner_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "cv_file_id": None,
                "cv_meta": None,
            }
            await db.sessions.insert_one(doc)
        else:
            # Enforce ownership if session is already owned
            if existing.get("user_id"):
                if not owner_id or existing.get("user_id") != owner_id:
                    raise HTTPException(status_code=403, detail="Forbidden: session not owned by user")
            else:
                # If session is anonymous and now we have a user, claim it
                if owner_id:
                    await db.sessions.update_one({"_id": existing.get("_id")}, {"$set": {"user_id": owner_id}})
        
        # Handle CV file upload into GridFS and materialize a temp file for tools that require a local path
        file_path = None  # path to a temp file we will pass to the orchestrator (if available)
        if cv_file and cv_file.filename:
            logger.info(f"Processing uploaded file: {cv_file.filename}")
            if not any(cv_file.filename.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt']):
                raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")
            contents = await cv_file.read()
            bucket = AsyncIOMotorGridFSBucket(get_db(), bucket_name="cv_files")
            gridfs_filename = f"{session_id}_{int(time.time())}_{cv_file.filename}"
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
            # Write to a temp file so MCP tool can read from disk
            try:
                suffix = Path(cv_file.filename).suffix or ".pdf"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(contents)
                tmp.flush()
                tmp.close()
                file_path = tmp.name
                logger.info(f"Materialized CV to temp file: {file_path}")
                # Schedule cleanup after response completes
                background_tasks.add_task(lambda p: (os.path.exists(p) and os.remove(p)), file_path)
            except Exception as e:
                logger.warning(f"Failed to create temp file for CV: {e}")
                file_path = None

        # Restore a previously uploaded CV for this session if available.
        # The LLM will decide whether to use it based on the user's current query.
        if not file_path:
            try:
                sess_doc = await db.sessions.find_one({"session_id": session_id})
                existing_file_id = (sess_doc or {}).get("cv_file_id")
                if existing_file_id:
                    bucket = AsyncIOMotorGridFSBucket(get_db(), bucket_name="cv_files")
                    # Download bytes from GridFS
                    stream = await bucket.open_download_stream(ObjectId(existing_file_id))
                    contents = await stream.read()
                    await stream.close()
                    # Use original filename suffix if known
                    suffix = ((sess_doc or {}).get("cv_meta", {}) or {}).get("original_filename", "cv.pdf")
                    suffix = Path(suffix).suffix or ".pdf"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(contents)
                    tmp.flush()
                    tmp.close()
                    file_path = tmp.name
                    logger.info(f"Restored CV from GridFS to temp file: {file_path}")
                    background_tasks.add_task(lambda p: (os.path.exists(p) and os.remove(p)), file_path)
            except Exception as e:
                logger.warning(f"Could not restore CV from GridFS: {e}")
        
        # Use user's provided input as-is; let the LLM infer intent and plan tools.
        unified_message = message or user_query
        
        # Build context for orchestrator
        # NOTE: Do NOT persist the user message yet. We will only persist once the LLM produced a response.
        
        # LLM-first Orchestrator streaming path (feature-flagged)
        if getattr(settings, "LLM_ORCHESTRATION_ENABLED", False) and llm_orchestrator:
            async def llm_stream_generator():
                assistant_buffer = ""
                try:
                    logger.info(f"Streaming LLM orchestrator for session {session_id}")
                    yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
                    # Load recent chat history (last 30 messages) for context-aware responses, scoped to user
                    chat_history: List[Dict[str, Any]] = []
                    try:
                        msg_query = {"session_id": session_id}
                        if owner_id:
                            msg_query["user_id"] = owner_id
                        cursor = db.messages.find(msg_query).sort("timestamp", 1)
                        messages_all = await cursor.to_list(length=1000)
                        # Keep last 30 to avoid prompt bloat
                        for m in messages_all[-30:]:
                            chat_history.append({
                                "role": m.get("role", "user"),
                                "content": (m.get("content") or "")[:2000],  # paranoia cap
                                "timestamp": m.get("timestamp"),
                            })
                    except Exception:
                        # Non-fatal; continue without history if it fails
                        chat_history = []

                    # CV availability is purely contextual; LLM decides when to use it.
                    allow_cv = bool(file_path)
                    ctx = {
                        "session_id": session_id,
                        "intent_hint": "",  # keep neutral; LLM infers intent from the message/history
                        "has_cv": allow_cv,
                        "file_path": str(file_path) if allow_cv and file_path else "",
                        "cv_text": (cv_text or "") if allow_cv else "",
                        "cv_file_id": (await db.sessions.find_one({"session_id": session_id})).get("cv_file_id") if (await db.sessions.find_one({"session_id": session_id})) else None,
                        "chat_history": chat_history,
                    }
                    async for line in llm_orchestrator.process_query_stream(unified_message, context=ctx):
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict) and data.get("type") == "final":
                                assistant_buffer = data.get("final_response", "")
                        except Exception:
                            pass
                        yield line
                except Exception as e:
                    logger.error(f"Error during LLM orchestrator streaming for {session_id}: {e}")
                    yield json.dumps({"success": False, "error": str(e)}) + "\n"
                finally:
                    # Persist conversation only if LLM produced an answer and user is authenticated
                    if assistant_buffer.strip() and owner_id:
                        try:
                            # Insert the triggering user message
                            if unified_message:
                                await db.messages.insert_one({
                                    "session_id": session_id,
                                    "user_id": owner_id,
                                    "role": "user",
                                    "content": unified_message,
                                    "timestamp": datetime.utcnow().isoformat(),
                                })
                            # Insert the assistant response
                            await db.messages.insert_one({
                                "session_id": session_id,
                                "user_id": owner_id,
                                "role": "assistant",
                                "content": assistant_buffer,
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                            await db.sessions.update_one({"session_id": session_id}, {"$set": {"updated_at": datetime.utcnow().isoformat()}})
                        except Exception:
                            logger.warning("Failed to persist conversation", exc_info=True)

            # Stream with headers that discourage buffering by proxies and servers
            return StreamingResponse(
                llm_stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",  # nginx
                    "Connection": "keep-alive",
                },
                background=background_tasks,
            )

        # If LLM orchestrator is disabled or failed to initialize, return 503
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "llm_orchestrator_unavailable",
                "message": "LLM orchestrator is disabled or unavailable. Set LLM_ORCHESTRATION_ENABLED=1 and restart the server.",
                "session_id": session_id,
            },
        )
    
    except HTTPException as http_err:
        logger.error(f"HTTP error in workflow processing: {http_err.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in workflow processing: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "workflow_type": workflow_type,
                "session_id": session_id,
                "message": "An unexpected error occurred"
            }
        )

@app.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """List sessions for the authenticated user."""
    db = get_db()
    user_id = current_user["_id"]
    cursor = db.sessions.find({"user_id": user_id}).sort("updated_at", -1)
    sessions = []
    async for s in cursor:
        # Count only this user's messages for safety
        msg_count = await db.messages.count_documents({"session_id": s.get("session_id"), "user_id": user_id})
        if msg_count == 0:
            # Skip sessions that have no persisted history
            continue
        sessions.append({
            "session_id": s.get("session_id"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
            "message_count": msg_count,
            "last_message_preview": "",
        })
    # Populate preview texts with last user message
    for item in sessions:
        # Fetch last user message for this user within the session
        last_user_msg = await db.messages.find({"session_id": item["session_id"], "role": "user", "user_id": user_id}).sort("timestamp", -1).to_list(1)
        if last_user_msg:
            content = last_user_msg[0].get("content") or ""
            item["last_message_preview"] = (content[:120] + ("..." if len(content) > 120 else "")) if content else ""
    return {"sessions": sessions}

@app.get("/chat/{session_id}")
async def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """Return full chat history for a session (DB-backed) enforcing ownership."""
    db = get_db()
    sess = await db.sessions.find_one({"session_id": session_id})
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Retrieve messages scoped to this user (defensive; ownership is already enforced)
    messages = await db.messages.find({"session_id": session_id, "user_id": current_user["_id"]}).sort("timestamp", 1).to_list(None)
    # Map to simple objects
    out_msgs = [{"role": m.get("role"), "content": m.get("content"), "timestamp": m.get("timestamp")} for m in messages]
    return {
        "session_id": session_id,
        "created_at": sess.get("created_at"),
        "updated_at": sess.get("updated_at"),
        "messages": out_msgs,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llm_ready = bool(llm_orchestrator)
    # Pull simple stats from MongoDB
    try:
        db = get_db()
        total_sessions = await db.sessions.count_documents({})
        total_messages = await db.messages.count_documents({})
    except Exception:
        total_sessions = None
        total_messages = None
    return {
        "status": "healthy" if llm_ready else "degraded",
        "app": "CV Processing API",
        "version": "1.0.0",
        "components": {
            "fastapi": "healthy",
            "llm_orchestrator": "healthy" if llm_ready else "unavailable",
            "mongodb": "ok" if total_sessions is not None else "unavailable",
        },
        "stats": {
            "sessions": total_sessions,
            "messages": total_messages,
        },
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": "CV Processing API",
        "version": "1.0.0",
        "description": "API for processing and analyzing CV/Resume documents using AI",
        "endpoints": {
            "workflow": "/workflow",
            "sessions": "/sessions",
            "chat": "/chat/{session_id}",
            "health": "/health",
            "auth": "/auth/*",
        },
        "frontend": "React application should be running on http://localhost:3000"
    }

def handle_exit(signum, frame):
    """Handle exit signals."""
    logger.info("Received shutdown signal")
    # Graceful shutdown is managed by FastAPI lifespan; just exit.
    # If needed, we can trigger orchestrator shutdown tasks here, but uvicorn will call lifespan shutdown.
    try:
        if llm_orchestrator:
            # Best-effort async shutdown trigger
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(llm_orchestrator.shutdown())
    except Exception:
        pass
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
