import asyncio
import json
import re
import logging
from typing import Any, AsyncGenerator, Dict, Optional, List
import argparse
import os
import sys

import httpx
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from config import settings
from cv_client import MCPClient as CVClient
from job_client import MCPClient as JobCrawlerClient

logger = logging.getLogger(__name__)


# --- Pydantic schemas for plan validation (top-level) ---
class ExecutionStep(BaseModel):
    tool: Literal["process_cv_pdf", "extract_jobs"]
    params: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None


class PlanSchema(BaseModel):
    workflow: Literal["cv_only", "job_only", "cv_job_matching"] = "cv_only"
    user_intent: Optional[str] = None
    execution_steps: List[ExecutionStep] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class GroqCVJobHost:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.base_url = settings.GROQ_API_URL.rstrip("/")  # e.g., https://api.groq.com/openai/v1
        self.api_key = settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.cv_client: Optional[CVClient] = None
        self.job_client: Optional[JobCrawlerClient] = None
        self.context: Dict[str, Any] = {}
        self.force_llm_planning: bool = True
        self.force_llm_final: bool = True

        # Always emit concise spinner-style statuses; verbose reasoning is suppressed

        # Planner prompt (same schema as simple.py, explicit params required)
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

        # Final response prompt
        self.synthesis_prompt = """You are a precise, reliable assistant. Generate the final output based on the WORKFLOW and provided artifacts. DO NOT hallucinate.

USER QUERY: {query}
WORKFLOW: {workflow}
TOOL RESULTS (truncated): {results}
CV_DATA (JSON): {cv_json}
CV_TEXT: {cv_text}
MARKDOWN_CONTENT (for job extraction): {markdown}

SCENARIOS:
1) When WORKFLOW == "cv_only":
   - Answer EXACTLY what the user asked. Do not add extra sections or suggestions unless explicitly requested.
   - Never ask the user follow-up questions (e.g., "What are your skills?").
   - If the user asks to summarize, output a single concise paragraph (3–5 sentences) capturing:
     • Professional background and seniority (if evident)
     • Core domains/roles and notable achievements
     • Key strengths/technologies/tools
     • Education/certifications if relevant
   - If the user asks for skills, extract only skills. If they ask a specific question, answer it directly.
   - Rely on CV_TEXT as the primary source. Use CV_DATA only to complement if helpful.
   - If information is not present in the CV, state briefly that it is not mentioned.
   - Do NOT output any code, pseudo-code, or JSON unless explicitly asked.

2) When WORKFLOW == "job_only":
   - Parse MARKDOWN_CONTENT and EXTRACT ALL job listings.
   - Return ONLY a JSON array. No commentary before or after.
   - Heuristics for finding jobs:
     * Titles in headings (# …) or emphasized text (**bold**, *italic*)
     * Sections with description, requirements, etc.
     * Separated by horizontal rules (---) or headers
     * Phrases: "Job Title", "Position", "Role", "Opening", "We're hiring", "Apply now"
   - OUTPUT FORMAT:
     [
       {{
         "title": "Exact job title",
         "company": "Company name",
         "location": "Work location or 'not mentioned'",
         "description": "Brief description of job duties",
         "salary": "Salary information or 'not mentioned'",
         "requirements": "Key qualifications or 'not mentioned'",
         "contrat-type": "Full-time, Part-time, etc. or 'not mentioned'",
         "required_skill": "Key skills needed or 'not mentioned'",
         "post_date": "When job was posted or 'not mentioned'"
       }}
     ]
   - IMPORTANT RULES:
     * Use "not mentioned" for missing fields (never null or N/A)
     * Extract ALL distinct job positions (do not limit)
     * Deduplicate identical job entries
     * One JSON object per job position
     * Be generous: if a job is mentioned, include it

3) When WORKFLOW == "cv_job_matching":
   - Keep existing behavior: compare CV_DATA against the extracted job requirements.
   - Highlight fit and gaps, matching skills/experiences, and missing qualifications.
   - Be concise, structured, and actionable.

IMPORTANT:
- Use only provided data (CV_DATA, MARKDOWN_CONTENT, TOOL RESULTS). If something is missing, state it briefly.
- Do not invent URLs or file paths. Do not add sections not asked for.

Your Output:"""

        # Split prompts per workflow for precise control
        self.cv_only_prompt = (
            """
You are a precise assistant. Based ONLY on the user's request and the CV contents, answer succinctly.

USER QUERY: {query}
CV_DATA (JSON, truncated): {cv_json}
CV_TEXT (truncated): {cv_text}

Rules:
- Answer exactly what was asked about the CV. No follow-up questions.
- If summarizing, provide a single concise paragraph (3–5 sentences) capturing background, core domains, strengths, notable achievements, and relevant education.
- If info is missing in the CV, state 'not mentioned' briefly.
- DO NOT output JSON unless explicitly asked.
"""
        ).strip()

        # Fallback prompt that returns an object with an items array, so we can enforce JSON object mode
        self.job_only_object_prompt = (
            """
You are given MARKDOWN content of a job listing page. Extract ALL distinct job postings and output ONLY a JSON object with key "items" whose value is an array of job objects.

MARKDOWN_CONTENT (truncated):
{markdown}

Output strictly this JSON object shape:
{{
  "items": [
    {{
      "title": "Exact job title",
      "company": "Company name",
      "location": "Work location or 'not mentioned'",
      "description": "Brief description of job duties",
      "salary": "Salary information or 'not mentioned'",
      "requirements": "Key qualifications or 'not mentioned'",
      "contrat-type": "Full-time, Part-time, etc. or 'not mentioned'",
      "required_skill": "Key skills needed or 'not mentioned'",
      "post_date": "When job was posted or 'not mentioned'"
    }}
  ]
}}

Rules:
- Use 'not mentioned' for missing fields.
- Deduplicate identical entries.
- No prose, ONLY the JSON object with an "items" array.
"""
        ).strip()

        self.job_only_prompt = (
            """
You are given MARKDOWN content of a job listing page. Extract ALL distinct job postings and output ONLY a JSON array.

MARKDOWN_CONTENT (truncated):
{markdown}

Output strictly this JSON array shape:
[
  {
    "title": "Exact job title",
    "company": "Company name",
    "location": "Work location or 'not mentioned'",
    "description": "Brief description of job duties",
    "salary": "Salary information or 'not mentioned'",
    "requirements": "Key qualifications or 'not mentioned'",
    "contrat-type": "Full-time, Part-time, etc. or 'not mentioned'",
    "required_skill": "Key skills needed or 'not mentioned'",
    "post_date": "When job was posted or 'not mentioned'"
  }
]

Rules:
- Use 'not mentioned' for missing fields.
- Deduplicate identical entries.
- No prose, ONLY the JSON array.
 - The very first character of your output MUST be '[' and the last character MUST be ']'.
"""
        ).strip()

    async def initialize(self):
        try:
            self.cv_client = CVClient(settings.CV_MCP_SERVER_SCRIPT)
            if hasattr(self.cv_client, "start"):
                await asyncio.wait_for(self.cv_client.start(), timeout=60)
            logger.info("✅ CV client initialized")
        except Exception as e:
            logger.warning(f"❌ CV client unavailable: {e}")
            self.cv_client = None
        try:
            self.job_client = JobCrawlerClient(settings.JOB_MCP_SERVER_SCRIPT)
            if hasattr(self.job_client, "start"):
                await asyncio.wait_for(self.job_client.start(), timeout=60)
            logger.info("✅ Job client initialized")
        except Exception as e:
            logger.warning(f"❌ Job client unavailable: {e}")
            self.job_client = None

    async def _groq_chat(self, messages: list[Dict[str, str]], *, max_tokens: int = 800, force_json: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        # Try to enforce JSON with response_format (OpenAI compatible). If unsupported, Groq ignores it.
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        # Cap timeout to 12s to align with planning hard deadline and avoid long hangs
        effective_timeout = min(getattr(settings, "GROQ_TIMEOUT", 60) or 60, 12)
        base = (self.base_url or "").rstrip("/")
        # Accept either a base API URL (…/v1) or a full endpoint (…/chat/completions)
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return text.strip()

    async def call_llm(self, prompt: str, max_tokens: int = 800, *, force_json: bool = False) -> str:
        messages = [
            {"role": "system", "content": "You are a precise planner. Output strictly follows instructions."},
            {"role": "user", "content": prompt},
        ]
        try:
            return await self._groq_chat(messages, max_tokens=max_tokens, force_json=force_json)
        except Exception as e:
            logger.error(f"Groq LLM error: {e}")
            return f"❌ LLM Error: {str(e)}"

    async def parse_json_response(self, text: str) -> Dict[str, Any]:
        def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
            # Map 'steps' -> 'execution_steps' if present
            if isinstance(obj, dict) and "execution_steps" not in obj and "steps" in obj:
                obj = dict(obj)
                obj["execution_steps"] = obj.get("steps") or []
            return obj

        def _validate(obj: Dict[str, Any]) -> Dict[str, Any]:
            try:
                model = PlanSchema.model_validate(_normalize(obj))
                return model.model_dump()
            except ValidationError as ve:
                return {
                    "workflow": "cv_only",
                    "execution_steps": [],
                    "confidence": 0.1,
                    "error": f"Plan validation failed: {ve.errors()}",
                }

        # First attempt: parse as-is
        try:
            obj = json.loads(text)
            return _validate(obj)
        except Exception:
            pass

        # Second attempt: extract JSON object from text
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                obj = json.loads(text[start:end])
                return _validate(obj)
        except Exception:
            pass

        return {
            "workflow": "cv_only",
            "execution_steps": [],
            "confidence": 0.1,
            "error": "Failed to parse or validate LLM response",
        }

    def _parse_jobs_array(self, text: str) -> List[Dict[str, Any]]:
        """Robustly parse a JSON array of jobs from text. Returns [] on failure.
        Accepts either a pure JSON array or text containing one JSON array.
        """
        try:
            if not text or not isinstance(text, str):
                return []
            # Strip common code fences
            t = text.strip()
            if t.startswith('```'):
                t = re.sub(r"^```[a-zA-Z]*\n?|```$", "", t).strip()
            trimmed = t
            # Direct array
            if trimmed.startswith('['):
                arr = json.loads(trimmed)
                return arr if isinstance(arr, list) else []
        except Exception:
            pass
        # Try to extract first [...] region
        try:
            start = trimmed.find('[')
            end = trimmed.rfind(']') + 1
            if start >= 0 and end > start:
                arr = json.loads(trimmed[start:end])
                return arr if isinstance(arr, list) else []
        except Exception:
            pass
        # Fallback: single object -> wrap into array
        try:
            if trimmed.startswith('{') and trimmed.endswith('}'):
                obj = json.loads(trimmed)
                return [obj] if isinstance(obj, dict) else []
        except Exception:
            pass
        # Fallback: recover multiple dicts by scanning balanced braces and wrapping into an array
        try:
            objs: List[Dict[str, Any]] = []
            depth = 0
            start_idx = -1
            for i, ch in enumerate(trimmed):
                if ch == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_idx != -1:
                            candidate = trimmed[start_idx:i+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    objs.append(obj)
                            except Exception:
                                pass
                            start_idx = -1
            if objs:
                return objs
        except Exception:
            pass
        # Final fallback A: if content looks like bare key-value lines (single block), wrap with { }
        try:
            if re.search(r'^\s*"[A-Za-z0-9_\- ]+"\s*:', trimmed):
                candidate = '{' + trimmed.strip().strip(',') + '}'
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return [obj]
        except Exception:
            pass
        # Final fallback B: group multiple objects from key-value lines separated by blank lines
        try:
            lines = [ln for ln in trimmed.splitlines()]
            blocks: list[str] = []
            buf: list[str] = []
            for ln in lines:
                if not ln.strip():
                    if buf:
                        blocks.append('\n'.join(buf))
                        buf = []
                    continue
                # Keep only lines that look like JSON key-value pairs
                if re.search(r'"[^"]+"\s*:', ln):
                    buf.append(ln.strip().rstrip(','))
            if buf:
                blocks.append('\n'.join(buf))
            objs: list[Dict[str, Any]] = []
            for b in blocks:
                try:
                    kv_lines = [l for l in b.splitlines() if ':' in l]
                    if not kv_lines:
                        continue
                    inner = ',\n'.join(kv_lines)
                    candidate = '{' + inner + '}'
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except Exception:
                    continue
            if objs:
                return objs
        except Exception:
            pass
        return []

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "process_cv_pdf" and self.cv_client:
                result = await self.cv_client.call_tool(tool_name, params)
                return {"success": True, "result": result, "tool": tool_name}
            elif tool_name in ["extract_jobs"] and self.job_client:
                result = await self.job_client.call_tool(tool_name, params)
                return {"success": True, "result": result, "tool": tool_name}
            else:
                return {"success": False, "error": f"Tool {tool_name} not available"}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Tool {tool_name} timeout"}
        except Exception as e:
            return {"success": False, "error": f"Tool {tool_name} failed: {str(e)}"}

    async def process_query_stream(self, user_query: str, *, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        ctx = context or {}

        # Pre-extract candidate URLs
        try:
            candidates: list[str] = []
            for m in re.findall(r"https?://\S+", user_query):
                url = m.rstrip('),].;!"\'')
                candidates.append(url)
            if candidates:
                ctx["candidate_urls"] = candidates
                if not ctx.get("job_url"):
                    ctx["job_url"] = candidates[0]
        except Exception:
            pass

        # Early config validation
        if not (self.api_key or "").strip():
            yield json.dumps({"type": "error", "message": "groq_api_key_missing"}) + "\n"
            yield json.dumps({"type": "final", "workflow": None, "tools_used": 0}) + "\n"
            return

        # Planning (always LLM-based for simplicity)
        yield json.dumps({"type": "progress", "stage": "planning", "percent": 5, "message": "Starting planning"}) + "\n"
        yield json.dumps({"type": "status", "message": "🤖 Planning with LLM..."}) + "\n"
        try:
            planner_prompt = self.workflow_prompt.format(query=user_query, context=json.dumps(ctx, ensure_ascii=False))
            raw_plan = await self.call_llm(planner_prompt, max_tokens=700, force_json=True)
            plan = await self.parse_json_response(raw_plan)
            workflow = plan.get("workflow") or ("job_only" if ctx.get("job_url") else ("cv_only" if ctx.get("file_path") else "cv_only"))
            steps = plan.get("execution_steps") or []
        except Exception as e:
            # Minimal fallback: infer simple workflow from context
            logger.warning(f"Planner failed, falling back: {e}")
            if ctx.get("file_path") and ctx.get("job_url"):
                workflow = "cv_job_matching"
                steps = [
                    {"tool": "process_cv_pdf", "params": {"file_path": ctx.get("file_path")}},
                    {"tool": "extract_jobs", "params": {"url": ctx.get("job_url"), "max_pages": 1}},
                ]
            elif ctx.get("job_url"):
                workflow = "job_only"
                steps = [{"tool": "extract_jobs", "params": {"url": ctx.get("job_url"), "max_pages": 1}}]
            elif ctx.get("file_path"):
                workflow = "cv_only"
                steps = [{"tool": "process_cv_pdf", "params": {"file_path": ctx.get("file_path")}}]
            else:
                workflow = "cv_only"
                steps = []
        # Planning phase complete
        try:
            yield json.dumps({"type": "progress", "stage": "planning", "percent": 15, "message": "Plan ready"}) + "\n"
        except Exception:
            pass

        # If no steps planned, provide a helpful response without executing tools
        if not steps:
            yield json.dumps({"type": "status", "message": "No execution steps planned"}) + "\n"
            # Craft a concise assistant response to guide the user
            guidance = (
                "I can analyze a CV PDF or extract jobs from a job URL. "
                "Please upload your CV (PDF) or include a job link in your message, and I'll proceed."
            )
            yield json.dumps({"type": "response", "content": guidance}) + "\n"
            yield json.dumps({"type": "final", "workflow": workflow, "tools_used": 0}) + "\n"
            return

        # Suppress plan summary emission; rely on spinner statuses

        # Execute planned steps
        all_results = []
        normalized_steps = steps[:3]
        total_tools = max(1, len(normalized_steps))
        tool_weight = 70 / total_tools  # distribute 70% across tools
        for i, step in enumerate(normalized_steps, 1):
            tool_name = step.get("tool")
            params = step.get("params", {})
            start_percent = int(15 + (i - 1) * tool_weight)
            yield json.dumps({"type": "progress", "stage": "tools", "percent": start_percent, "message": f"Running {tool_name}"}) + "\n"
            yield json.dumps({"type": "status", "message": f"⏳ Running {tool_name}..."}) + "\n"
            result = await self.call_tool(tool_name, params)
            all_results.append({"step": i, "tool": tool_name, "result": result})
            if tool_name == "extract_jobs" and result.get("success"):
                payload_outer = result.get("result", {}) or {}
                payload = payload_outer.get("result", payload_outer) or {}
                combined_md = payload.get("combined_markdown", "")
                if combined_md:
                    ctx["markdown_content"] = combined_md
                # Capture structured jobs if provided by the tool
                try:
                    jobs = payload.get("jobs") or payload.get("extracted_jobs") or payload.get("items")
                    if isinstance(jobs, list):
                        ctx["jobs_json"] = jobs
                except Exception:
                    pass
                # Propagate job_url used to improve final synthesis messaging
                try:
                    url = params.get("url") or params.get("start_url")
                    if url:
                        ctx["job_url"] = url
                except Exception:
                    pass
            elif tool_name == "process_cv_pdf" and result.get("success"):
                payload_outer = result.get("result", {}) or {}
                payload = payload_outer.get("result", payload_outer) or {}
                ctx["cv_data"] = payload
                try:
                    if isinstance(payload, dict):
                        ctx["cv_text"] = payload.get("text") or payload.get("corrected_text") or ""
                    # Suppress detailed CV capture status to keep steps minimal
                except Exception:
                    pass
            yield json.dumps({"type": "tool_complete", "tool": tool_name, "success": result.get("success", False)}) + "\n"
            done_percent = int(15 + i * tool_weight)
            yield json.dumps({"type": "progress", "stage": "tools", "percent": min(85, done_percent), "message": f"Finished {tool_name}"}) + "\n"

        # Final synthesis
        # If job workflow but neither markdown nor structured jobs captured, end early with a helpful message
        if workflow == "job_only" and ctx.get("job_url") and not ctx.get("markdown_content") and not ctx.get("jobs_json"):
            yield json.dumps({"type": "status", "message": "⚠️ Could not fetch/convert the page to markdown and no jobs found."}) + "\n"
            yield json.dumps({"type": "response", "content": "I couldn't retrieve readable content from the provided URL, and no jobs were extracted. Please verify the link is accessible and try again."}) + "\n"
            yield json.dumps({"type": "final", "workflow": workflow, "tools_used": len(normalized_steps)}) + "\n"
            return

        # Synthesis stage status
        yield json.dumps({"type": "progress", "stage": "synthesis", "percent": 90, "message": "Generating answer"}) + "\n"
        yield json.dumps({"type": "status", "message": "⏳ Generating answer..."}) + "\n"

        # Sanitize large payloads to avoid HTTP 413; tailor per workflow
        if workflow == "cv_only":
            safe_results = ""
            safe_cv_json = "{}"
            safe_cv_text = (ctx.get("cv_text") or "")[:8000]
            safe_markdown = ""
            token_cap = 500
        else:
            if workflow == "job_only":
                safe_results = "[]"
            else:
                safe_results = json.dumps(all_results, ensure_ascii=False)[:1500]

            cv_payload = ctx.get("cv_data") or {}
            if isinstance(cv_payload, dict):
                for key in ["raw_text", "full_text", "pages", "content", "text"]:
                    try:
                        if key in cv_payload and isinstance(cv_payload[key], str):
                            cv_payload[key] = cv_payload[key][:4000]
                    except Exception:
                        pass
            safe_cv_json = json.dumps(cv_payload, ensure_ascii=False)[:6000]
            safe_cv_text = (ctx.get("cv_text") or "")[:6000]

            if workflow == "job_only":
                # Allow larger markdown so chunking can extract more jobs
                safe_markdown = (ctx.get("markdown_content") or "")[:36000]
            elif workflow == "cv_job_matching":
                safe_markdown = (ctx.get("markdown_content") or "")[:9000]
            else:
                safe_markdown = ""

            if workflow == "job_only":
                token_cap = 300
            elif workflow == "cv_job_matching":
                token_cap = 550
            else:
                token_cap = 600

            # Secondary clamp if combined input still large
            try:
                combined_len = len(safe_cv_json) + len(safe_cv_text) + len(safe_markdown) + len(safe_results)
                if workflow == "cv_job_matching" and combined_len > 18000:
                    # Prefer reducing markdown and cv_text further
                    safe_markdown = safe_markdown[:6000]
                    safe_cv_text = safe_cv_text[:4000]
                    # Keep some structure in cv_json but clamp
                    safe_cv_json = safe_cv_json[:5000]
            except Exception:
                pass

        # Final call via helper
        try:
            final_text = await self.generate_final_response(
                workflow=workflow,
                cv_json=safe_cv_json,
                cv_text=safe_cv_text,
                markdown=safe_markdown,
                all_results=safe_results,
                token_cap=token_cap,
                query=user_query,
                context=ctx,
            )
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"synthesis_error: {str(e)}"}) + "\n"
            yield json.dumps({"type": "final", "workflow": workflow, "tools_used": len(normalized_steps)}) + "\n"
            return

        # Stream the final response in chunks for better UX
        try:
            text = final_text or ""
            chunk_size = 200
            for i in range(0, len(text), chunk_size):
                part = text[i:i+chunk_size]
                if part:
                    yield json.dumps({"type": "response", "content": part}) + "\n"
                    await asyncio.sleep(0)
        except Exception:
            yield json.dumps({"type": "response", "content": final_text}) + "\n"
        yield json.dumps({"type": "progress", "stage": "done", "percent": 100, "message": "Completed"}) + "\n"
        yield json.dumps({"type": "final", "workflow": workflow, "tools_used": len(normalized_steps)}) + "\n"

    async def generate_final_response(
        self,
        *,
        workflow: str,
        cv_json: str,
        cv_text: str,
        markdown: str,
        all_results: str,
        token_cap: int,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """Generate the final response per workflow.

        - job_only: prefer direct passthrough of structured jobs captured in context["jobs_json"].
          If absent, and markdown exists, call LLM with job_only_prompt to parse. Else return empty array.
        - cv_only: use cv_only_prompt with cv_json/cv_text.
        - cv_job_matching/other: use consolidated synthesis_prompt with all inputs.
        """
        try:
            if workflow == "job_only":
                jobs = context.get("jobs_json")
                if not self.force_llm_final and isinstance(jobs, list) and jobs:
                    return json.dumps(jobs, ensure_ascii=False)
                if markdown and markdown.strip():
                    # Chunked parsing: split markdown into manageable pieces to avoid truncation
                    md = markdown
                    chunk_size = 9000  # chars per chunk
                    chunks = [md[i:i+chunk_size] for i in range(0, len(md), chunk_size)] or [md]
                    merged: list[dict] = []
                    seen_keys: set[str] = set()
                    # Use a more generous cap per chunk to allow multiple items
                    per_chunk_tokens = max(token_cap, 1600)
                    for ch in chunks:
                        try:
                            prompt = self.job_only_prompt.format(markdown=ch)
                            out = await self.call_llm(prompt, max_tokens=per_chunk_tokens)
                            arr = self._parse_jobs_array(out)
                            if isinstance(arr, list):
                                for j in arr:
                                    if not isinstance(j, dict):
                                        continue
                                    title = str(j.get("title") or j.get("job_title") or j.get("position") or "").strip().lower()
                                    company = str(j.get("company") or j.get("employer") or "").strip().lower()
                                    location = str(j.get("location") or "").strip().lower()
                                    # Build a stable key for dedupe
                                    key = f"{title}|{company}|{location}"
                                    if key in seen_keys:
                                        continue
                                    seen_keys.add(key)
                                    merged.append(j)
                        except Exception as e:
                            logger.warning(f"job_only chunk parse failed: {e}")
                            continue
                    if merged:
                        return json.dumps(merged, ensure_ascii=False)
                    # Fallback: if tool provided structured jobs, return them
                    tool_jobs = context.get("jobs_json")
                    if isinstance(tool_jobs, list) and tool_jobs:
                        return json.dumps(tool_jobs, ensure_ascii=False)
                    # Last resort: ask model for an object with items[] using forced JSON mode
                    try:
                        obj_prompt = self.job_only_object_prompt.format(markdown=markdown)
                        obj_text = await self.call_llm(obj_prompt, max_tokens=max(1200, token_cap), force_json=True)
                        # Try direct parse
                        data = json.loads(obj_text)
                        items = data.get("items") if isinstance(data, dict) else None
                        if isinstance(items, list):
                            return json.dumps(items, ensure_ascii=False)
                        # Try to extract object region and parse
                        s = obj_text.find('{')
                        e = obj_text.rfind('}') + 1
                        if s >= 0 and e > s:
                            data = json.loads(obj_text[s:e])
                            items = data.get("items") if isinstance(data, dict) else None
                            if isinstance(items, list):
                                return json.dumps(items, ensure_ascii=False)
                    except Exception:
                        pass
                return "[]"
        except Exception as e:
            logger.warning(f"job_only final generation fallback failed: {e}")

        if workflow == "cv_only":
            prompt = self.cv_only_prompt.format(query=query, cv_json=cv_json, cv_text=cv_text)
            return await self.call_llm(prompt, max_tokens=token_cap)

        # Combined/default synthesis
        prompt = self.synthesis_prompt.format(
            query=query,
            workflow=workflow,
            results=all_results,
            cv_json=cv_json,
            cv_text=cv_text,
            markdown=markdown,
        )
        return await self.call_llm(prompt, max_tokens=token_cap)

    async def shutdown(self):
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


if __name__ == "__main__":
    # Minimal CLI to test Groq planning and streaming without FastAPI
    parser = argparse.ArgumentParser(description="Test GroqCVJobHost planning stream")
    parser.add_argument("--message", "-m", required=True, help="User message/query")
    parser.add_argument("--job-url", help="Optional job URL to include in context")
    parser.add_argument("--cv", help="Optional path to a CV PDF to include in context")
    parser.add_argument("--model", help="Override Groq model (defaults to settings or class default)")

    args = parser.parse_args()

    async def _run():
        model = args.model or None
        host = GroqCVJobHost(model=model) if model else GroqCVJobHost()
        await host.initialize()

        ctx: Dict[str, Any] = {}
        if args.job_url:
            ctx["job_url"] = args.job_url
        if args.cv and os.path.exists(args.cv):
            ctx["file_path"] = args.cv

        print("--- Streaming NDJSON ---", flush=True)
        try:
            async for line in host.process_query_stream(args.message, context=ctx):
                # Lines already include trailing \n
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception as e:
            # Surface unexpected errors in NDJSON format for consistency
            sys.stdout.write(json.dumps({"type": "error", "message": f"cli_stream_error: {str(e)}"}) + "\n")
            sys.stdout.flush()
        finally:
            await host.shutdown()

    asyncio.run(_run())
