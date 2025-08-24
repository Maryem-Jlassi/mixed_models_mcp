import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
import re

import httpx
from config import settings
from cv_client import MCPClient as CVClient
from job_client import MCPClient as JobCrawlerClient

logger = logging.getLogger(__name__)


class SimplifiedCVJobHost:
    def __init__(self, model: str = "llama3.2:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model
        self.cv_client: Optional[CVClient] = None
        self.job_client: Optional[JobCrawlerClient] = None
        self.context: Dict[str, Any] = {}

        # Single LLM call for workflow planning
        self.workflow_prompt = """Given the user query and context, determine what tools to run.

QUERY: {query}
CONTEXT: {context}

TOOLS:
- process_cv_pdf: Extract text from CV PDF (needs file_path)
- extract_jobs: Get jobs from URL (needs url, optional max_pages)

OUTPUT FORMAT (JSON only):
{{
  "steps": [
    {{"tool": "tool_name", "params": {{"param": "value"}}}}
  ]
}}

RULES:
1. If context has file_path and query mentions CV: use process_cv_pdf
2. If query has URL or context has job_url: use extract_jobs  
3. Max 2 steps
4. Use exact parameter names: file_path, url, max_pages
5. Strict parameter validity:
   - process_cv_pdf: require an existing file_path from CONTEXT (do NOT invent or guess; if not present, do not include this step).
   - extract_jobs: require a non-empty url (from CONTEXT.job_url or detected in QUERY). If absent, do not include this step.
6. Set workflow consistently with steps:
   - If plan includes only process_cv_pdf -> workflow = "cv_only"
   - If plan includes only job tools (extract_jobs, deep_crawl_job_pages, fetch_url_html_with_pagination, html_to_markdown) -> workflow = "job_only"
   - If plan includes both CV and job tools -> workflow = "cv_job_matching"

Examples:
Query: "analyze my CV" + context has file_path="/tmp/cv.pdf"
{{"workflow": "cv_only", "confidence": 0.6, "steps": [{{"tool": "process_cv_pdf", "params": {{"file_path": "/tmp/cv.pdf"}}}}]}}

Query: "get jobs from https://example.com"
{{"workflow": "job_only", "confidence": 0.6, "steps": [{{"tool": "extract_jobs", "params": {{"url": "https://example.com", "max_pages": 1}}}}]}}
"""

        # Separate prompts per workflow to reduce cross-contamination
        self.cv_only_prompt = """You are analyzing a CV.

USER QUERY: {query}
CV_DATA: {cv_json}
CV_TEXT: {cv_text}

INSTRUCTIONS:
- Answer EXACTLY what the user asked. Do not add extra sections or advice unless explicitly requested.
- If the user asks ask a specific question, answer it directly without extra informations
- Rely on CV_TEXT as the primary source; use CV_DATA only to complement if helpful.
- If information is not present in the CV, state briefly that it is not mentioned.
- No code and no JSON in the output.
- Keep it concise, professional, and specific, referencing CV content when useful.

STRICT RULES:
- Never ask the user questions unless they asked you to ask back. Do not respond with follow-up questions like "What are your skills?".
- When the query includes words like "summarize", "summary", or similar, output a single concise paragraph (3–5 sentences) capturing:
  • Professional background and years/level (if evident)
  • Core domains/roles and notable achievements
  • Key strengths/technologies/tools
  • Education/certifications if relevant
  Keep it neutral, factual, and derived strictly from the CV.
"""

        self.job_only_prompt = """Extract ALL job listings from the markdown content. Return ONLY a JSON array.

MARKDOWN_CONTENT: {markdown}

Look for job indicators:
- Headings with job titles
- Job descriptions and requirements
- Company names and locations

OUTPUT FORMAT (JSON array only):
[
  {{
    "title": "Job title",
    "company": "Company name or 'not mentioned'",
    "location": "Location or 'not mentioned'",
    "description": "Brief description or 'not mentioned'",
    "salary": "Salary info or 'not mentioned'",
    "requirements": "Requirements or 'not mentioned'",
    "contrat-type": "Contract type or 'not mentioned'",
    "required_skill": "Skills or 'not mentioned'",
    "post_date": "Date or 'not mentioned'"
  }}
]

Return ONLY the JSON array, no other text."""

        self.cv_job_matching_prompt = """You are a strict evaluator. Determine if the candidate CV aligns with the job.

USER QUERY: {query}
CV_DATA (JSON): {cv_json}
JOB PAGE (Markdown): {markdown}

TASK:
- Use ONLY information present in CV_DATA or JOB PAGE. No assumptions.
- If multiple jobs appear, select the single most relevant one; otherwise use the main job.
- Do NOT rewrite or improve the job post. Focus on CV vs job match.
- Respond in the same language as the USER QUERY.

OUTPUT (plain text, concise, use the following headings exactly; for each claim include a short CV evidence in quotes):
Verdict: <Yes | Partial | No> — Match score: <0-100>
Key matches:
- <bullet 1> (CV evidence: "<short quote or field from CV>")
- <bullet 2> (CV evidence: "<short quote or field from CV>")
Gaps:
- <bullet 1> (CV evidence: "not mentioned" or "<quote showing it's different>")
Experience fit:
- <role/years from CV mapped to job needs> (CV evidence: "<role, dates, metrics>")
Language fit:
- <language level from CV vs job requirement> (CV evidence: "<language line from CV>" or "not mentioned")
Missing requirements:
- <any required certs/skills not found in CV> (CV evidence: "not mentioned")
Recommendations:
- <1–3 short, actionable steps to close gaps>
Short pitch:
- <2–3 sentences the candidate could use as a tailored pitch>

STRICT RULES:
- If a fact is not in the CV or page, say "non mentionné" / "not mentioned".
- Prefer concrete evidence from CV (projects, roles, metrics). Keep to 8–12 lines total.
 - Do NOT treat salary, benefits, hours, or HR policy as matching criteria; you may mention them only as informational and they must not affect the score.
"""

    async def get_synthesis_response(self, user_query: str, workflow: str, ctx: Dict[str, Any]) -> str:
        """Get workflow-specific response using targeted prompts."""
        if workflow == "cv_only":
            cv_data = ctx.get("cv_data", {})
            safe_cv_json = json.dumps(cv_data, ensure_ascii=False)[:40000]
            cv_text = (ctx.get("cv_text") or cv_data.get("text") or cv_data.get("corrected_text") or "")
            safe_cv_text = cv_text[:20000]
            prompt = self.cv_only_prompt.format(query=user_query, cv_json=safe_cv_json, cv_text=safe_cv_text)
            return await self.call_llm(prompt, max_tokens=600)
        elif workflow == "job_only":
            markdown = (ctx.get("markdown_content") or "")[:15000]
            prompt = self.job_only_prompt.format(markdown=markdown)
            return await self.call_llm(prompt, max_tokens=800, force_json=True)
        elif workflow == "cv_job_matching":
            cv_data = ctx.get("cv_data", {})
            markdown = (ctx.get("markdown_content") or "")[:20000]
            safe_cv_json = json.dumps(cv_data, ensure_ascii=False)[:30000]
            prompt = self.cv_job_matching_prompt.format(query=user_query, cv_json=safe_cv_json, markdown=markdown)
            return await self.call_llm(prompt, max_tokens=900)
        return "Unknown workflow"

    async def initialize(self):
        """Initialize MCP clients"""
        try:
            self.cv_client = CVClient("cv_ocr_server.py")
            if hasattr(self.cv_client, "start"):
                await asyncio.wait_for(self.cv_client.start(), timeout=60)
            logger.info("✅ CV client initialized")
        except Exception as e:
            logger.warning(f"❌ CV client unavailable: {e}")
            self.cv_client = None

        try:
            self.job_client = JobCrawlerClient("server.py")  
            if hasattr(self.job_client, "start"):
                await asyncio.wait_for(self.job_client.start(), timeout=60)
            logger.info("✅ Job client initialized")
        except Exception as e:
            logger.warning(f"❌ Job client unavailable: {e}")
            self.job_client = None

    async def call_llm(self, prompt: str, max_tokens: int = 800, *, force_json: bool = False) -> str:
        """Single LLM call method"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        # Enforce JSON when requested regardless of prompt wording
                        "format": "json" if force_json or ("STRICT JSON" in prompt) else None,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return f"❌ LLM Error: {str(e)}"

    async def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Robust JSON parsing with normalization for 'steps' alias."""
        def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
            # Map 'steps' -> 'execution_steps' if needed
            if isinstance(obj, dict) and "execution_steps" not in obj and "steps" in obj:
                obj = dict(obj)  # shallow copy
                obj["execution_steps"] = obj.get("steps") or []
            return obj

        try:
            # Try direct parsing first
            obj = json.loads(text)
            return _normalize(obj)
        except Exception:
            # Extract JSON from text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start:end])
                    return _normalize(obj)
                except Exception:
                    pass
        
        # Fallback structure
        return {
            "workflow": "cv_only",
            "execution_steps": [],
            "confidence": 0.1,
            "error": "Failed to parse LLM response"
        }

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool"""
        try:
            # Route to appropriate client
            if tool_name == "process_cv_pdf" and self.cv_client:
                # No timeout: let the MCP tool run to completion
                result = await self.cv_client.call_tool(tool_name, params)
                return {"success": True, "result": result, "tool": tool_name}
                
            elif tool_name in ["extract_jobs"] and self.job_client:
                # No timeout: let the MCP tool run to completion
                result = await self.job_client.call_tool(tool_name, params)
                return {"success": True, "result": result, "tool": tool_name}
            else:
                return {"success": False, "error": f"Tool {tool_name} not available"}
                
        except asyncio.TimeoutError:
            # Shouldn't happen now, but keep for safety
            return {"success": False, "error": f"Tool {tool_name} timeout"}
        except Exception as e:
            return {"success": False, "error": f"Tool {tool_name} failed: {str(e)}"}

    async def process_query_stream(self, user_query: str, *, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Main processing method - only 2 LLM calls total"""
        ctx = context or {}

        # Lightweight pre-extraction of candidate URLs for the planner (non-routing hint only)
        try:
            candidates: list[str] = []
            for m in re.findall(r"https?://\S+", user_query):
                url = m.rstrip('),].;!"\'')
                candidates.append(url)
            if candidates:
                ctx["candidate_urls"] = candidates
                # If job_url not already set by backend, default to first detected
                if not ctx.get("job_url"):
                    ctx["job_url"] = candidates[0]
        except Exception:
            pass
        
        # Step 1: Plan workflow. Deterministic routing based on available context to avoid planner bias
        yield json.dumps({"type": "progress", "stage": "planning", "percent": 5, "message": "Starting planning"}) + "\n"
        yield json.dumps({"type": "status", "message": "⏳ Planning..."}) + "\n"
        plan = None
        if ctx.get("file_path") and ctx.get("job_url"):
            # Deterministic: both present => matching
            workflow = "cv_job_matching"
            steps = [
                {"tool": "process_cv_pdf", "params": {"file_path": ctx.get("file_path")}},
                {"tool": "extract_jobs", "params": {"url": ctx.get("job_url"), "max_pages": 1}},
            ]
        elif ctx.get("job_url") and not ctx.get("file_path"):
            # Deterministic: only job URL => job_only
            workflow = "job_only"
            steps = [
                {"tool": "extract_jobs", "params": {"url": ctx.get("job_url"), "max_pages": 1}},
            ]
        elif ctx.get("file_path") and not ctx.get("job_url"):
            # Deterministic: only CV => cv_only
            workflow = "cv_only"
            steps = [{"tool": "process_cv_pdf", "params": {"file_path": ctx.get("file_path")}}]
        else:
            try:
                workflow_response = await asyncio.wait_for(
                    self.call_llm(
                        self.workflow_prompt.format(
                            query=user_query,
                            context=json.dumps(ctx, ensure_ascii=False)
                        ),
                        max_tokens=150,  # Smaller cap for faster planning
                        force_json=True
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                yield json.dumps({"type": "error", "message": "planning_timeout"}) + "\n"
                yield json.dumps({"type": "final", "workflow": None, "tools_used": 0}) + "\n"
                return
            except Exception as e:
                try:
                    yield json.dumps({"type": "error", "message": f"planning_error: {str(e)}"}) + "\n"
                except Exception:
                    pass
                yield json.dumps({"type": "final", "workflow": None, "tools_used": 0}) + "\n"
                return
            
            plan = await self.parse_json_response(workflow_response)
            workflow = plan.get("workflow", "cv_only")
            steps = plan.get("execution_steps", [])

        # If the LLM included a URL in the planned steps, propagate it into context for downstream logic.
        try:
            for s in steps:
                if isinstance(s, dict) and s.get("tool") == "extract_jobs":
                    params = s.get("params", {})
                    if not ctx.get("job_url") and params.get("url"):
                        ctx["job_url"] = params.get("url")
        except Exception:
            pass

        # Announce end of planning phase
        yield json.dumps({"type": "progress", "stage": "planning", "percent": 15, "message": "Plan ready"}) + "\n"
        
        # No fallback: if planner produced no steps, surface and stop without executing tools
        if not steps:
            yield json.dumps({"type": "status", "message": "No execution steps planned"}) + "\n"
            yield json.dumps({"type": "final", "workflow": workflow, "tools_used": 0}) + "\n"
            return

        # Allow the LLM to decide workflow without forcing job_only heuristics

        yield json.dumps({
            "type": "plan", 
            "workflow": workflow, 
            "steps": len(steps)
        }) + "\n"
        
        # Step 2: Execute tools directly (NO extra heuristics)
        all_results = []
        
        # Execute exactly the steps planned by the LLM (cap to 3)
        normalized_steps = steps[:3]
        total_tools = max(1, len(normalized_steps))
        tool_weight = 70 / total_tools  # distribute 70% across tools

        attempted_inline_fetch = False
        for i, step in enumerate(normalized_steps[:3], 1):  # Max 3 tools
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            start_percent = int(15 + (i - 1) * tool_weight)
            yield json.dumps({"type": "progress", "stage": "tools", "percent": start_percent, "message": f"Running {tool_name}"}) + "\n"
            yield json.dumps({"type": "status", "message": f"⏳ Running {tool_name}..."}) + "\n"
            
            # No special inline refetch logic is needed anymore since we use extract_jobs

            # Execute tool
            result = await self.call_tool(tool_name, params)
            all_results.append({
                "step": i,
                "tool": tool_name,
                "result": result
            })
            
            # Store important results in context
            if tool_name == "extract_jobs" and result.get("success"):
                # Unwrap nested structure: our call_tool() wraps job_client.call_tool() which already wraps the server result
                payload_outer = result.get("result", {}) or {}
                payload = payload_outer.get("result", payload_outer) or {}
                try:
                    combined_md = payload.get("combined_markdown", "")
                    if combined_md:
                        ctx["markdown_content"] = combined_md
                        ctx["extract_jobs_meta"] = {
                            "pages_fetched": payload.get("pages_fetched"),
                            "pages_with_markdown": payload.get("pages_with_markdown"),
                            "combined_markdown_length": payload.get("combined_markdown_length"),
                        }
                    else:
                        ctx["extract_jobs_meta"] = {
                            "pages_fetched": payload.get("pages_fetched"),
                            "pages_with_markdown": payload.get("pages_with_markdown"),
                            "combined_markdown_length": payload.get("combined_markdown_length"),
                            "first_page_status": (payload.get("pages") or [{}])[0].get("status") if payload.get("pages") else None,
                            "first_page_markdown_length": (payload.get("pages") or [{}])[0].get("markdown_length") if payload.get("pages") else None,
                            "first_page_error": (payload.get("pages") or [{}])[0].get("error") if payload.get("pages") else None,
                        }
                except Exception:
                    pass
            elif tool_name == "process_cv_pdf" and result.get("success"):
                # Unwrap nested structure like with extract_jobs
                payload_outer = result.get("result", {}) or {}
                payload = payload_outer.get("result", payload_outer) or {}
                ctx["cv_data"] = payload
                try:
                    if isinstance(payload, dict):
                        ctx["cv_text"] = payload.get("text") or payload.get("corrected_text") or ""
                except Exception:
                    pass
                # Suppress detailed CV capture status to keep steps minimal
                
            yield json.dumps({
                "type": "tool_complete",
                "tool": tool_name, 
                "success": result.get("success", False)
            }) + "\n"
            done_percent = int(15 + i * tool_weight)
            yield json.dumps({"type": "progress", "stage": "tools", "percent": min(85, done_percent), "message": f"Finished {tool_name}"}) + "\n"

        # Step 3: Single LLM call for final synthesis (COMPREHENSIVE)
        # Guard: if URL-based extraction was attempted and no markdown present, avoid hallucinations.
        if ctx.get("job_url") and not ctx.get("markdown_content"):
            # Surface diagnostics gathered from extract_jobs to help debugging
            diag = ctx.get("extract_jobs_meta") or {}
            yield json.dumps({
                "type": "status",
                "message": "⚠️ Could not fetch/convert the page to markdown. Skipping AI summary.",
                "diagnostics": diag,
            }) + "\n"
            yield json.dumps({"type": "response", "content": "I couldn't retrieve readable content from the provided URL. Please verify the link is accessible and try again."}) + "\n"
            yield json.dumps({"type": "final", "workflow": workflow, "tools_used": len(normalized_steps)}) + "\n"
            return
        yield json.dumps({"type": "progress", "stage": "synthesis", "percent": 90, "message": "Generating answer"}) + "\n"
        yield json.dumps({"type": "status", "message": "⏳ Generating answer..."}) + "\n"

        # Sanitize payload to keep Ollama prompt small
        if workflow == "job_only":
            safe_results = "[]"
        else:
            safe_results = json.dumps(all_results, ensure_ascii=False)[:2000]

        cv_payload = ctx.get("cv_data") or {}
        if isinstance(cv_payload, dict):
            for key in ["raw_text", "full_text", "pages", "content", "text"]:
                if key in cv_payload:
                    cv_payload[key] = None
        safe_cv_json = json.dumps(cv_payload, ensure_ascii=False)[:40000]

        if workflow == "job_only":
            safe_markdown = (ctx.get("markdown_content") or "")[:15000]
        elif workflow == "cv_job_matching":
            safe_markdown = (ctx.get("markdown_content") or "")[:20000]
        else:
            safe_markdown = ""

        # Use workflow-specific prompts to generate the final response
        final_response = await self.get_synthesis_response(user_query, workflow, ctx)

        # Stream the final response in chunks for better UX
        try:
            text = final_response or ""
            chunk_size = 200
            for i in range(0, len(text), chunk_size):
                part = text[i:i+chunk_size]
                if part:
                    yield json.dumps({"type": "response", "content": part}) + "\n"
                    await asyncio.sleep(0)  # yield control to flush progressively
        except Exception:
            # Fallback to single-shot if chunking fails
            yield json.dumps({"type": "response", "content": final_response}) + "\n"
        # Completion markers
        yield json.dumps({"type": "progress", "stage": "done", "percent": 100, "message": "Completed"}) + "\n"
        yield json.dumps({"type": "final", "workflow": workflow, "tools_used": len(normalized_steps)}) + "\n"

    async def shutdown(self):
        """Cleanup resources"""
        try:
            if self.cv_client and hasattr(self.cv_client, "stop"):
                await self.cv_client.stop()
        except Exception:
            pass
        try:
            if self.job_client and hasattr(self.job_client, "stop"):
                await self.job_client.stop()
        except Exception:
            pass


# Example usage (optional manual test)
async def main():
    host = SimplifiedCVJobHost()
    await host.initialize()
    
    # Single quick demo: Job URL only (avoids heavy CV OCR)
    ctx = {"job_url": "https://www.keejob.com/offres-emploi/"}
    query = "Find recent offers from this URL"

    print(f"\n🔵 Test: {query}")
    print(f"📄 Context: {ctx}")

    async for chunk in host.process_query_stream(query, context=ctx):
        try:
            data = json.loads(chunk.strip())
            if data.get("type") == "response":
                print(f"🤖 {data['content']}")
            elif data.get("type") == "status":
                print(f"⏳ {data['message']}")
            elif data.get("type") == "error":
                print(f"❌ {data['message']}")
        except Exception:
            continue
    
    await host.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
