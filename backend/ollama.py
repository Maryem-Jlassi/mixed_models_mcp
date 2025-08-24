import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import async_timeout
import httpx
from pydantic import BaseModel, Field, validator, ValidationError

from config import settings
from cv_client import MCPClient as CVClient
from job_client import MCPClient as JobCrawlerClient

logger = logging.getLogger(__name__)


class ThinkingPhase(Enum):
    INITIAL_ANALYSIS = "initial_analysis"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    EXECUTION = "execution"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    FINAL_SYNTHESIS = "final_synthesis"


@dataclass
class ThinkingStep:
    phase: ThinkingPhase
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningPrompts:
    PLANNING = """<thinking>
    Based on the analysis and available tools, I need to create a detailed execution plan.
    
    ANALYSIS:
    {analysis}
    
    AVAILABLE TOOLS:
    {available_tools}
    
    I will now create a step-by-step execution plan that:
    1. Addresses the user's intent from the analysis
    2. Uses the most appropriate tools in the right order
    3. Handles any dependencies between steps
    4. Includes error handling and fallbacks
    </thinking>
    
    Respond with STRICT JSON only (no markdown, no additional text). Shape MUST match PlanSchema and PlanStepSchema exactly:
    {{
        "execution_steps": [
            {{
                "step": 1,
                "tool_category": "cv_processing|job_search",
                "rationale": "why this step is needed",
                "expected_inputs": {{
                    "file_path": "<from context when processing CV>",
                    "url": "<from context when processing job URL>"
                }},
                "expected_outputs": "what the tool should return",
                "dependencies": []
            }}
        ],
        "estimated_duration": "time estimate",
        "potential_issues": ["list", "of", "potential", "issues"]
    }}"""
    
    MERGED_ANALYSIS_AND_PLAN = """<thinking>
    I need to analyze the user query and create an optimized execution plan in a single step to minimize API calls.

    USER QUERY: {query}
    AVAILABLE TOOLS: CV Processing, Job Search/Crawling
    AVAILABLE CONTEXT: {context}

    TOOL DESCRIPTIONS (with exact schemas):
    1) fetch_url_html_with_pagination
       - purpose: fetch raw HTML from a URL, possibly across pages

    2) html_to_markdown
       - purpose: convert HTML to Markdown

    3) process_cv_pdf
       - purpose: extract corrected text and stats from a provided CV PDF file

    RULES:
1. ALWAYS produce at least 1 execution step. Do NOT return an empty plan.
2. If a CV is available in context, ONLY include CV-related steps when the user's current query explicitly asks for CV analysis, CV parsing, or CV-job matching. Otherwise, ignore CV context even if present.
3. If CV corrected text exists in context (e.g., corrected_text/previous_corrected_text), prefer using it and avoid reprocessing the file.
4. MAX 2-3 STEPS TOTAL. Choose the shortest valid path; avoid loops/retries unless strictly necessary.
5. Combine operations where possible (e.g., fetch + markdown in one step). If HTML is already available, proceed to final synthesis immediately (you may pass HTML directly to synthesis if Markdown is not required).
6. Skip discovery phase if URL is provided.
7. Prefer direct URL access over crawling; only crawl if absolutely necessary.
8. ORDERED EXECUTION: fetch_html (limit pages) → (optional) html_to_markdown → final synthesis. Stop as soon as sufficient HTML/Markdown is available.
9. PARAMETER STRICTNESS: when specifying tool parameters, use EXACT key names and types from the schemas above. Do NOT invent or rename keys.

SCENARIOS:
1. DIRECT URL:
   - fetch_html → html_to_markdown
   - No precheck step. No deep crawl.

2. COMPANY NAME ONLY (no URL provided):
   - Do NOT attempt discovery. Ask the user to provide a job or careers URL.

3. CV PROCESSING (user asks for CV analysis):
   - If corrected text is already in context, proceed to synthesis; otherwise run process_cv_pdf(file_path=context.file_path)
   - Do NOT use job search tools unless the user asked for matching to jobs.

4. CV PRESENT IN CONTEXT (but user did not ask for CV-related help):
   - Ignore CV context for this turn and focus on the user's request (e.g., job search or general chat).

Respond with STRICT JSON only (no markdown, no additional text). Ensure execution_steps is NON-EMPTY when a CV is present in context. Each execution step MUST follow this shape to match the system schema (PlanStepSchema):
{{
    "query_type": "cv_processing|job_search|combined|unclear",
    "user_intent": "brief description of user goal",
    "execution_steps": [
        {{
            "step": 1,
            "action": "tool_name",
            "tool_category": "cv_processing|job_search",
            "rationale": "why this step is needed",
            "expected_inputs": {{}},
            "expected_outputs": "what this step will accomplish",
            "dependencies": []
        }}
    ],
    "confidence": 0.0-1.0,
    "requires_cv": true|false
}}

EXAMPLE 1 - Direct URL:
{{
    "query_type": "job_search",
    "user_intent": "extract job details from the given URL",
    "execution_steps": [
        {{
            "step": 1,
            "action": "fetch_url_html_with_pagination",
            "tool_category": "job_search",
            "rationale": "fetch the job page HTML",
            "expected_inputs": {{"url": "<from context.job_url>", "max_pages": 1}},
            "expected_outputs": "Get HTML content of the job page",
            "dependencies": []
        }},
        {{
            "step": 2,
            "action": "html_to_markdown",
            "tool_category": "job_search",
            "rationale": "convert HTML to Markdown",
            "expected_inputs": {{"html_content": "<from step 1 HTML>"}},
            "expected_outputs": "Markdown representation of the job page",
            "dependencies": ["1"]
        }}
    ],
    "confidence": 0.95,
    "requires_cv": false
}}

EXAMPLE 2 - CV Analysis (only when user explicitly asks for CV analysis):
{{
    "query_type": "cv_processing",
    "user_intent": "analyze and summarize CV",
    "execution_steps": [
        {{
            "step": 1,
            "action": "process_cv_pdf",
            "tool_category": "cv_processing",
            "rationale": "extract corrected text and stats from the uploaded CV",
            "expected_inputs": {{"file_path": "<from context.file_path>"}},
            "expected_outputs": "Corrected CV text and parsing statistics",
            "dependencies": []
        }}
    ],
    "confidence": 0.9,
    "requires_cv": true
}}
"""
    
    TOOL_SELECTION = """<thinking>
I need to select the most appropriate tool for the current step from the dynamically discovered tools.

Current Step: {current_step}
Available Tools (discovered): {available_tools}
Context: {context}

SCENARIO PLAYBOOK (map user intent to tool and exact parameters):
- If context contains job_url (a direct job or careers URL):
  - Use fetch_url_html_with_pagination .
- If you already have raw HTML from a previous step:
  - Use html_to_markdown FROM_PREVIOUS_STEP.
- If the user provides no URL (company name only):
  - Do NOT select any job tool. Ask the user to provide a job or careers URL in the final response.
- CV-only queries (no job intent, context has file_path):
  - Use process_cv_pdf .

FEW-SHOT EXAMPLES (tool selection JSON only):
- Query: "Here is a job link: https://example.com/jobs/123"
  Context: {{"job_url": "https://example.com/jobs/123"}}
  Output:
  {{"selected_tool": "fetch_url_html_with_pagination", "tool_parameters": {{"url": "https://example.com/jobs/123", "max_pages": 1}}, "selection_rationale": "URL present; fetch HTML", "expected_output": "HTML content of the page", "confidence": 0.9}}

- Query: "Analyze this HTML I already have"
  Context: {{"html": "<html>...</html>"}}
  Output:
  {{"selected_tool": "html_to_markdown", "tool_parameters": {{"html_content": "<html>...</html>"}}, "selection_rationale": "Convert provided HTML to Markdown", "expected_output": "Markdown string", "confidence": 0.85}}

- Query: "Analyze my CV"
  Context: {{"file_path": "/tmp/user_cv.pdf"}}
  Output:
  {{"selected_tool": "process_cv_pdf", "tool_parameters": {{"file_path": "/tmp/user_cv.pdf"}}, "selection_rationale": "CV-only intent", "expected_output": "CV summary/skills", "confidence": 0.9}}

Exact SCHEMAS to respect:
- fetch_url_html_with_pagination: {{"url": string, "max_pages"?: integer=1}}
- html_to_markdown: {{"html_content": string}}

Decision rules (choose only one best fit):
- Use fetch_url_html_with_pagination when you have a specific URL and need its HTML (possibly across pages).
- Use html_to_markdown right after you have HTML content to prepare Markdown for synthesis.
- If no URL is available, do NOT select any job tool; request a URL from the user in the final response.

MANDATORY CONSTRAINTS (must follow exactly):
- If Context contains "job_url":
  - You MUST select fetch_url_html_with_pagination and pass {{"url": Context.job_url}}.
- NEVER output placeholders (e.g., "<url>") and NEVER omit required parameters.

Before selecting, validate REQUIRED params are available. If a required param is missing, either choose a different tool or extract it from the user's query (e.g., infer company_name like "extract jobs from X" -> company_name = "X"). Do NOT invent URLs or company names.

Parameter requirements (critical):
- Always provide a complete tool_parameters object with concrete values (no placeholders).
- Use EXACT key names and types per schemas above.
- If the current step is CV-related and context.file_path exists, set tool_parameters.file_path = context.file_path.
 - If a URL is present in the user's query or context, map it exactly to:
  - fetch_url_html_with_pagination.url
 - For html_to_markdown, set tool_parameters.html_content = "<HTML_FROM_PREVIOUS_STEP>".
- NEVER change or guess domains; use exactly the URL found.

Minimize iterations:
- Prefer the shortest valid path. If HTML is already present in context (or from a previous step), do NOT refetch/crawl; proceed to final synthesis.
- If Markdown is not necessary, pass HTML evidence directly to synthesis.

When the step is CV-related, prefer process_cv_pdf if context.file_path is available.

INVALID EXAMPLE (do not do this):
{{"selected_tool": "fetch_url_html_with_pagination", "tool_parameters": {{}}, "selection_rationale": "fetch without URL"}}
Reason: Missing required parameter url. This selection MUST be rejected.

Select the best tool from the discovered tools in strictly valid JSON only. Use double quotes for all keys and string values. No comments, no trailing commas, no additional prose or markdown. Output only the JSON object. Ensure tool_parameters are fully specified with concrete values (never placeholders, never missing required fields). If a required parameter is not available, select a different tool instead of emitting an invalid call:
{{
    "selected_tool": "exact_tool_name_from_available_tools",
    "tool_parameters": {{
        "parameter_name": "parameter_value"
    }},
    "selection_rationale": "detailed explanation of why this tool is the best choice",
    "expected_output": "what I expect this tool to return",
    "parameter_mapping": "how I mapped the step requirements to tool parameters",
    "fallback_options": ["alternative_tool_names_if_primary_fails"],
    "confidence": 0.0-1.0
}}
"""

    EXECUTION_RESULT = """<thinking>
I need to analyze the execution result and determine next steps.

Tool Used: {tool_name}
Parameters: {parameters}
Result: {result}
Success: {success}

Let me evaluate:
1. Did the tool execute successfully?
2. Is the output what I expected?
3. Do I have enough information to proceed?
4. Are there any errors or issues to address?
5. What should be the next step?

Important rule: If sufficient context for final synthesis is already available (for example, Markdown evidence exists in context under key 'markdown_evidence'), set next_action to "complete" rather than retrying more tools.
Additional rule: If raw HTML content is available and Markdown is not strictly required, proceed to final synthesis using HTML (set next_action = "complete"). Avoid additional iterations.
</thinking>

Analyze the execution result in strictly valid JSON only. Use double quotes for all keys and string values. No comments, no trailing commas, no additional prose or markdown. Output only the JSON object:
{{
    "execution_success": true/false,
    "output_quality": "excellent|good|acceptable|poor",
    "key_findings": ["important", "insights", "from", "result"],
    "issues_detected": ["any", "problems", "found"],
    "next_tool": "next_tool_name",
    "next_tool_parameters": {"param": "value"},
    "justification": "why this is the right next step given the result",
    "confidence": 0.0-1.0
}}
"""

    REFLECTION = """<thinking>
Let me reflect on the entire process so far.

{{ ... }}
    "confidence": 0.0-1.0
}}
"""

    FINAL_SYNTHESIS = """<thinking>
I must deliver a precise, user-centered answer STRICTLY grounded in the extracted evidence.
 No generic site descriptions. No invented details.

Original Query: {original_query}

Available Evidence (truncated where needed):
- All tool results: {all_results}
- Key insights: {key_insights}
- Tool-extracted jobs (ground truth for listing items): {jobs_found}
- Evidence priority: Markdown if present; otherwise raw HTML is acceptable.
- Evidence sources may include: precheck signals, HTML content, Markdown content, CV summary (ONLY if requested), or extracted job objects.

Ground rules:
1) Use CV content ONLY if the user's current query explicitly asks for CV analysis or CV-job matching. If not requested, ignore CV context even if available.
2) Be brief, factual, and actionable. Prefer bullet points. Include URLs that are directly relevant.
3) If no relevant evidence was obtained, state that clearly and propose one concrete next step.
4) You may NOT conclude “no jobs found” unless at least one HTML fetch and markdown conversion attempt was made. If only a precheck occurred, recommend proceeding to fetch and parsing instead of finalizing.

Format:
- A short title
- A crisp summary grounded in evidence
- If single job posting: output sections
  • Title
  • Company
  • Location
  • Job Type
  • Responsibilities (3-6 bullets)
  • Requirements (3-6 bullets)
  • Benefits (optional)
  • How to apply (with grounded URL if available)

- If job listing page: output sections
  • Summary (counts only if supported by evidence)
  • Top Opportunities (3-5) – each with Title, Company, Location (if any), and grounded URL
  • Notes/Instructions – only if explicitly present in evidence

Safety & Grounding Rules:
- Never speculate. Never fabricate company names, locations, or URLs.
- Keep numbers and counts only when directly extractable.
- If conflicting info appears, mention the conflict with both variants.

Failure Handling:
- If precheck/fetch/crawl failed or evidence is empty: state what failed and suggest providing a specific job URL or reducing scope.

End the answer with a short actionable next step (e.g., "Share another company or a specific job URL to go deeper").
"""

    EXECUTION_RESULT = """<thinking>
I need to analyze the execution result and determine next steps.

Tool Used: {tool_name}
Parameters: {parameters}
Result: {result}
Success: {success}

Let me evaluate:
1. Did the tool execute successfully?
2. Is the output what I expected?
3. Do I have enough information to proceed?
4. Are there any errors or issues to address?
5. What should be the next step?

Important rule: If sufficient context for final synthesis is already available (for example, Markdown evidence exists in context under key 'markdown_evidence'), set next_action to "complete" rather than retrying more tools.
Additional rule: If raw HTML content is available and Markdown is not strictly required, proceed to final synthesis using HTML (set next_action = "complete"). Avoid additional iterations.

Provide a concise, structured next step decision with justification.

RULES:
1. If the tool failed due to transient errors (timeouts, HTTP 5xx), retry ONCE with a small backoff or pick a safer alternative.
2. If the current step produced sufficient evidence (e.g., HTML or Markdown content), proceed directly to FINAL SYNTHESIS without extra calls.
3. Prefer minimal steps; avoid redundant crawling if HTML is already available.
4. Only conclude “no jobs found” AFTER at least one successful HTML fetch and Markdown conversion attempt has been made and parsing yields zero jobs.

Output strictly valid JSON:
{{
    "decision": "proceed|retry|finalize",
    "next_tool": "next_tool_name",
    "next_tool_parameters": {{"param": "value"}},
    "justification": "why this is the right next step given the result",
    "confidence": 0.0-1.0
}}
"""

    # System prompt applied when CV analysis is requested by the user
    CV_ASSISTANT_SYSTEM = """
You are a CV Analysis Assistant.

Strict rules:
- Only engage in CV-related steps when the user's current query asks for CV analysis.
- Plans with empty execution_steps are INVALID. Always produce at least one step.
- If the user asked for CV analysis and corrected text exists, proceed directly to synthesis; otherwise include a step to run process_cv_pdf with expected_inputs.file_path = context.file_path.
- Do not invent paths or placeholders; use concrete values taken from context.
- Keep plans concise (1 step for pure CV analysis).

Observation rules:
- If corrected_text or previous_corrected_text exists in context, prefer next_action = "complete" for synthesis rather than re-processing.
{{ ... }}

Safety:
- Never hallucinate profile details; only report what tools extracted.
"""

    # System prompt for final CV summary (used in streamed synthesis)
    CV_FINAL_SUMMARY = """
You are a precise CV summarization assistant.

Instructions:
- Read the input JSON (fields: original_query, cv_only, tool_results).
- From tool_results, extract any of the following if present: corrected_text, text, content, summary, skills, entities.
- Use the extracted CV content to produce a concise, well-structured summary. Do not invent details not present in the CV.
- Prefer bullet points and short sentences. Be specific and actionable.
- If multiple versions of text exist, prefer corrected_text over raw OCR text.
- If no usable text is found, say so and suggest re-uploading a clearer PDF.

Output format:
- Title: short line, e.g., "CV Summary"
- Profile summary (2-4 bullets)
- Key skills (bullet list)
- Experience highlights (3-6 bullets with role, company, brief impact; include years when present)
- Education (1-2 bullets)
- Notable achievements (optional)
- Suggested improvements (2-4 bullets for ATS/clarity/quant metrics)

Rules:
- Do not include any internal reasoning or JSON; output plain text only.
- Do not speculate. Only use information present in tool_results.
- Keep the total output concise.
"""


    # Unified workflow system for planning and tool selection
    WORKFLOW_SYSTEM = """
You are a Workflow Manager.

Strict rules:
- Use CV_ASSISTANT_SYSTEM only when the user's query is CV-related.
- Use JOB_SEARCH_ASSISTANT_SYSTEM when a job URL is provided or job_search intent is detected from the user's query.
- Do NOT include CV steps merely because a CV exists in context. Only include when requested by the user's current query.

Observation rules:
- If markdown_evidence already exists, prefer next_action = "complete" to proceed to synthesis.
- Only continue crawling when needed (e.g., listing pages) and within safe bounds.

Safety:
- Do not follow external links beyond the configured crawler.
- Do not summarize beyond what is present in markdown evidence or extracted jobs.
"""


class AnalysisSchema(BaseModel):
    query_type: str
    user_intent: str
    required_tools: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class PlanStepSchema(BaseModel):
    step: int
    action: Optional[str] = None
    tool_category: Optional[str] = None
    rationale: Optional[str] = None
    expected_inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: Optional[str] = None
    dependencies: List[Union[str, int]] = Field(default_factory=list)


class PlanSchema(BaseModel):
    execution_steps: List[PlanStepSchema] = Field(default_factory=list)
    estimated_duration: Optional[str] = None
    potential_issues: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class ToolSelectionSchema(BaseModel):
    selected_tool: str
    tool_parameters: Dict[str, Any] = Field(default_factory=dict)
    selection_rationale: Optional[str] = None
    expected_output: Optional[str] = None
    parameter_mapping: Optional[str] = None
    fallback_options: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class ExecutionObservationSchema(BaseModel):
    execution_success: bool
    output_quality: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    issues_detected: List[str] = Field(default_factory=list)
    next_action: str
    confidence: float = 0.0


class ReflectionSchema(BaseModel):
    query_satisfaction: Optional[str] = None
    result_completeness: Optional[str] = None
    additional_value: List[str] = Field(default_factory=list)
    process_improvements: List[str] = Field(default_factory=list)
    user_next_steps: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class OllamaLLMClient:
    """Ollama client with local model integration."""

    # Model configurations for Ollama models
    MODEL_CONFIGS = {
        "llama3.2:latest": {
            "max_tokens": 8000,
            "priority": 1,
            "context_length": 131072,
            "temperature_range": (0.1, 2.0),
            "best_for": ["general_purpose", "reasoning", "analysis"]
        }
    }

    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        timeout: float = 120.0,  # Longer timeout for local inference
        loop=None
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = float(timeout)
        self._loop = loop or asyncio.get_event_loop()
        
        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"Content-Type": "application/json"}
        )
        
    async def _check_ollama_health(self):
        """Check if Ollama is accessible and model is available."""
        try:
            # First check if Ollama is running
            async with async_timeout.timeout(10):
                response = await self._client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if self.model not in available_models:
                    logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                    return False, f"Model {self.model} not available. Available: {', '.join(available_models)}"
                
                logger.info(f"✅ Ollama is accessible with model: {self.model}")
                return True, None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API health check failed with HTTP {e.response.status_code}")
            return False, f"Ollama API error: HTTP {e.response.status_code}"
            
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.error(f"Ollama API health check failed: {type(e).__name__} - {e}")
            return False, f"Ollama connection error: {type(e).__name__}"

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """Generate text using Ollama API."""
        
        # Enforce single local model
        selected_model = "llama3.2:latest"
        _ = self.MODEL_CONFIGS["llama3.2:latest"]

        # Compute endpoint
        base = getattr(settings, "OLLAMA_API_URL", self.base_url)
        if isinstance(base, str) and base.rstrip("/").endswith("/api/chat"):
            endpoint = base.rstrip("/")
        else:
            endpoint = f"{self.base_url}/api/chat"

        # Compute safe max tokens with cap
        default_max = int(getattr(settings, "LLM_MAX_TOKENS", 800))
        hard_cap = int(getattr(settings, "OLLAMA_MAX_TOKENS_CAP", 1024))
        eff_max = int(max(1, min(int(max_tokens) if max_tokens is not None else default_max, hard_cap)))

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                # Stability-oriented defaults for local runs
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_batch": 1,
            },
        }
        if system_prompt:
            body["messages"].append({"role": "system", "content": system_prompt})
        body["messages"].append({"role": "user", "content": prompt})
        # Ollama uses num_predict as generation cap
        body["options"]["num_predict"] = eff_max

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= max_retries:
            try:
                async with async_timeout.timeout(self.timeout):
                    resp = await self._client.post(endpoint, json=body)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})
                content = msg.get("content", "")
                return content or ""
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.RequestError, asyncio.TimeoutError) as e:
                last_err = e
                # Log helpful diagnostics on server/client errors
                try:
                    if isinstance(e, httpx.HTTPStatusError) and getattr(e, "response", None) is not None:
                        resp_text = e.response.text or ""
                        snippet = resp_text[:800]
                        logger.warning(
                            "Ollama HTTP error %s – response snippet: %s | prompt_len=%d",
                            getattr(e.response, "status_code", "unknown"),
                            snippet.replace("\n", " "),
                            len(prompt or ""),
                        )
                except Exception:
                    pass
                if attempt < max_retries:
                    delay = min(5.0, 1.0 * (2 ** attempt))
                    logger.warning(f"Ollama request failed (attempt {attempt+1}/{max_retries}). Retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                logger.error(f"Ollama generate failed after {attempt+1} attempts: {e}")
                raise

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream token deltas from Ollama chat API.

        Yields incremental content strings as they arrive.
        """
        # Enforce single local model
        selected_model = "llama3.2:latest"
        _ = self.MODEL_CONFIGS[selected_model]

        base = getattr(settings, "OLLAMA_API_URL", self.base_url)
        if isinstance(base, str) and base.rstrip("/").endswith("/api/chat"):
            endpoint = base.rstrip("/")
        else:
            endpoint = f"{self.base_url}/api/chat"

        default_max = int(getattr(settings, "LLM_MAX_TOKENS", 800))
        hard_cap = int(getattr(settings, "OLLAMA_MAX_TOKENS_CAP", 1024))
        eff_max = int(max(1, min(int(max_tokens) if max_tokens is not None else default_max, hard_cap)))

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [],
            "stream": True,
            "options": {
                "temperature": float(temperature),
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_batch": 1,
                "num_predict": eff_max,
            },
        }
        if system_prompt:
            body["messages"].append({"role": "system", "content": system_prompt})
        body["messages"].append({"role": "user", "content": prompt})

        try:
            async with async_timeout.timeout(self.timeout):
                async with self._client.stream("POST", endpoint, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        if data.get("done"):
                            break
                        msg = data.get("message", {})
                        delta = msg.get("content") or ""
                        if delta:
                            yield delta
        except Exception as e:
            logger.warning(f"Ollama stream error: {e}")
            # Fall back to non-streaming
            try:
                text = await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    model=model or self.model,
                    temperature=temperature,
                )
                if text:
                    yield text
            except Exception:
                return

    async def generate_with_reasoning(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        force: bool = False,
        **_: Any,
    ) -> str:
        """Compatibility wrapper used by `llm_orchestrator.py`.

        For local Ollama we don't do special reasoning modes; we simply call
        `generate()` with the provided parameters and return the text.
        """
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            model=model or self.model,
            temperature=temperature,
        )

    async def aclose(self):
        try:
            await self._client.aclose()
        except Exception:
            pass

    def _select_best_model(self, prompt_length: int) -> tuple[str, dict]:
        """Select the best model given the prompt length."""
        # Always use the single local model
        selected_config = self.MODEL_CONFIGS["llama3.2:latest"]
        return "llama3.2:latest", selected_config

# NOTE: Orchestrator wiring will be handled in main.py using this OllamaLLMClient

class LLMOrchestratorOllama:
    """Standalone orchestrator that uses local Ollama client and MCP tools.

    Exposes initialize(), process_query_stream(), and shutdown() used by main.py.
    """

    def __init__(self, model: str = "llama3.2:latest"):
        self.llm = OllamaLLMClient(
            model=model,
            timeout=getattr(settings, "OLLAMA_TIMEOUT", 60),
        )
        self.cv_client: Optional[CVClient] = None
        self.job_client: Optional[JobCrawlerClient] = None
        self.thinking_history: List[Any] = []
        self.context: Dict[str, Any] = {}

    async def initialize(self):
        try:
            # Initialize MCP clients (with discovery guarded by timeout)
            self.cv_client = CVClient(getattr(settings, "CV_MCP_SERVER_SCRIPT", "cv_ocr_server.py"))
            if hasattr(self.cv_client, "start"):
                await asyncio.wait_for(self.cv_client.start(), timeout=getattr(settings, "MCP_TOOL_TIMEOUT", 90))
        except Exception:
            logger.warning("CV MCP client unavailable", exc_info=True)
            self.cv_client = None
        try:
            self.job_client = JobCrawlerClient(getattr(settings, "JOB_MCP_SERVER_SCRIPT", "server.py"))
            if hasattr(self.job_client, "start"):
                await asyncio.wait_for(self.job_client.start(), timeout=getattr(settings, "MCP_TOOL_TIMEOUT", 90))
        except Exception:
            logger.warning("Job MCP client unavailable", exc_info=True)
            self.job_client = None

    async def shutdown(self):
        # Best-effort close MCP clients and Ollama HTTP client
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
        try:
            await self.llm.aclose()
        except Exception:
            pass

    # ---------- Helpers (parsing, validation, events) ----------
    async def _emit_step(self, phase: ThinkingPhase, content: str, confidence: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        step = {
            "type": "thinking_step",
            "phase": phase.value,
            "content": content,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.thinking_history.append(step)
        return step

    async def _parse_json(self, text: str) -> Dict[str, Any]:
        # Borrow robust parsing from llm_orchestrator.py
        if not text or not isinstance(text, str):
            return {"error": "Empty or invalid input", "raw": str(text)[:500]}
        text = text.strip()
        if "```json" in text:
            try:
                s = text.find("```json") + 7
                e = text.find("```", s)
                if e > s:
                    seg = text[s:e].strip()
                    if seg:
                        return json.loads(seg)
            except Exception:
                pass
        # Brace-matching extraction
        try:
            start = text.find('{')
            if start >= 0:
                brace_count, end, in_string, escape = 1, start + 1, False, False
                while end < len(text) and brace_count > 0:
                    ch = text[end]
                    if not in_string:
                        if ch == '{' and not escape:
                            brace_count += 1
                        elif ch == '}' and not escape:
                            brace_count -= 1
                        elif ch == '"' and not escape:
                            in_string = True
                    else:
                        if ch == '"' and not escape:
                            in_string = False
                        elif ch == '\\' and not escape:
                            escape = True
                        else:
                            escape = False
                    end += 1
                if brace_count == 0:
                    js = text[start:end]
                    try:
                        return json.loads(js)
                    except json.JSONDecodeError:
                        js = re.sub(r',\s*([}\]])', r'\1', js)
                        js = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', js)
                        return json.loads(js)
        except Exception:
            pass
        # Regex fallback
        try:
            m = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        # Direct parse or literal_eval
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import ast
                cleaned = text.strip()
                if cleaned.startswith("```") and cleaned.endswith("```"):
                    cleaned = cleaned.strip("`")
                cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
                obj = ast.literal_eval(cleaned)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list):
                    return {"items": obj}
            except Exception:
                pass
        return {"error": "Failed to parse JSON response", "snippet": text[:500]}

    def _validate_schema(self, model_cls, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return model_cls.model_validate(data).model_dump()
        except ValidationError as ve:
            out = dict(data) if isinstance(data, dict) else {"raw": str(data)}
            out["validation_errors"] = [
                {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
                for e in ve.errors()
            ]
            return out

    async def _discover_all_tools(self) -> Dict[str, Any]:
        discovered = {
            "cv_processing": {"tools": [], "schemas": {}, "client": "cv_client"},
            "job_search": {"tools": [], "schemas": {}, "client": "job_client"},
        }
        # CV tools
        if self.cv_client:
            try:
                await asyncio.wait_for(self.cv_client.discover_tools(), timeout=getattr(settings, "MCP_TOOL_TIMEOUT", 90))
                for t in self.cv_client.tools:
                    discovered["cv_processing"]["tools"].append({
                        "name": t.name,
                        "description": getattr(t, "description", "") or getattr(t, "title", ""),
                        "schema": getattr(t, "inputSchema", {}),
                    })
            except Exception as e:
                logger.error(f"Discover CV tools failed: {e}")
        # Job tools
        if self.job_client:
            try:
                await asyncio.wait_for(self.job_client.discover_tools(), timeout=getattr(settings, "MCP_TOOL_TIMEOUT", 90))
                for t in self.job_client.tools:
                    discovered["job_search"]["tools"].append({
                        "name": t.name,
                        "description": getattr(t, "description", "") or getattr(t, "title", ""),
                        "schema": getattr(t, "inputSchema", {}),
                    })
            except Exception as e:
                logger.error(f"Discover Job tools failed: {e}")
        return discovered

    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        timeout = getattr(settings, "MCP_TOOL_TIMEOUT", 90)
        # Decide client by known categories or discovered sets
        client = None
        # Route strictly by discovered membership; no hardcoded fallbacks
        if self.cv_client and any(tool_name == t.name for t in getattr(self.cv_client, 'tools', [])):
            client = self.cv_client
        elif self.job_client and any(tool_name == t.name for t in getattr(self.job_client, 'tools', [])):
            client = self.job_client
        else:
            return {"success": False, "error": "tool_unavailable", "tool": tool_name}
        if client is None:
            return {"success": False, "error": "no_client_available"}
        try:
            # Emit lightweight debug about routing choice via return payload on error
            return await asyncio.wait_for(client.call_tool(tool_name, params), timeout=timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": "tool_timeout", "tool": tool_name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}

    # ---------- Main streaming workflow ----------
    async def process_query_stream(self, user_query: str, *, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        ctx = context or {}
        has_cv = bool(ctx.get("file_path") or ctx.get("cv_text"))
        cv_only = bool(has_cv )
        # Debug: emit cv flags to the stream
        try:
            yield json.dumps({"type": "status", "message": f"Context flags: has_cv={has_cv}, cv_only={cv_only}"}) + "\n"
        except Exception:
            pass
        # Discover tools
        discovered = await self._discover_all_tools()
        yield json.dumps(await self._emit_step(ThinkingPhase.INITIAL_ANALYSIS, f"Query received. has_cv={has_cv}", 0.6, {"context_keys": list(ctx.keys())})) + "\n"

        # Build merged analysis+plan prompt
        system_prompt = ReasoningPrompts.WORKFLOW_SYSTEM
        prompt = ReasoningPrompts.MERGED_ANALYSIS_AND_PLAN.format(query=user_query, context=json.dumps(ctx, ensure_ascii=False),)
        try:
            raw = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=getattr(settings, "LLM_MAX_TOKENS", 800))
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"LLM error: {e}"}) + "\n"
            return
        plan_obj = await self._parse_json(raw)
        plan = self._validate_schema(PlanSchema, plan_obj) if isinstance(plan_obj, dict) else {"execution_steps": []}
        yield json.dumps(await self._emit_step(ThinkingPhase.PLANNING, json.dumps(plan, ensure_ascii=False)[:800])) + "\n"

        # Ensure at least one step
        steps = plan.get("execution_steps") or []
        # Do not inject steps implicitly; rely on LLM plan only
        if not steps:
            # No implicit default steps. If planning produced no steps, proceed to synthesis or ask user for required inputs.
            plan["execution_steps"] = []

        all_results: List[Dict[str, Any]] = []
        for i, step in enumerate(steps[:3], start=1):
            current_step = {
                "step": step.get("step", i),
                "action": step.get("action"),
                "tool_category": step.get("tool_category"),
                "expected_inputs": step.get("expected_inputs", {}),
            }
            yield json.dumps(await self._emit_step(ThinkingPhase.TOOL_SELECTION, json.dumps(current_step))) + "\n"

            # Tool selection prompt
            # In CV-only mode, restrict to CV tools regardless of accidental category
            category = "cv_processing" if cv_only else (current_step.get("tool_category") or "job_search")
            available_tools = discovered.get(category, {}).get("tools", [])
            selection_prompt = ReasoningPrompts.TOOL_SELECTION.format(
                current_step=json.dumps(current_step, ensure_ascii=False),
                available_tools=json.dumps(available_tools, ensure_ascii=False),
                context=json.dumps(ctx, ensure_ascii=False),
            )
            try:
                selection_raw = await self.llm.generate(prompt=selection_prompt, system_prompt=None, temperature=0.2, max_tokens=400)
            except Exception as e:
                yield json.dumps({"type": "error", "message": f"LLM selection error: {e}"}) + "\n"
                break
            selection_obj = await self._parse_json(selection_raw)
            selection = self._validate_schema(ToolSelectionSchema, selection_obj) if isinstance(selection_obj, dict) else {}

            tool_name = selection.get("selected_tool") or current_step.get("action")
            tool_params = selection.get("tool_parameters") or current_step.get("expected_inputs") or {}
            # Debug: emit selected tool and parameter keys
            try:
                yield json.dumps({
                    "type": "status",
                    "message": f"Selected tool: {tool_name} | param_keys: {sorted(list(tool_params.keys()))}"
                }) + "\n"
            except Exception:
                pass
            # No implicit context parameter mapping; rely on LLM-provided parameters


            # No CV-only hard guard; follow LLM selection strictly
            # Emit status with timing hint before executing tool
            est = getattr(settings, "MCP_TOOL_TIMEOUT", 90)
            yield json.dumps({"type": "status", "message": f"Running {tool_name} ... this may take up to {est}s on first run"}) + "\n"
            yield json.dumps(await self._emit_step(ThinkingPhase.EXECUTION, json.dumps({"tool": tool_name, "params": tool_params})[:800], 0.8)) + "\n"
            _t0 = time.time()
            result = await self._call_tool(tool_name, tool_params)
            all_results.append({"tool": tool_name, "result": result})
            _dt = int(time.time() - _t0)
            yield json.dumps({"type": "status", "message": f"{tool_name} completed in {_dt}s"}) + "\n"

            # Capture markdown evidence when available from html_to_markdown tool
            try:
                if tool_name == "html_to_markdown" and result and result.get("success"):
                    # Result structure can vary; attempt common shapes safely
                    payload = result.get("result") or {}
                    markdown = None
                    if isinstance(payload, dict):
                        markdown = payload.get("markdown") or payload.get("content") or payload.get("text")
                    elif isinstance(payload, str):
                        markdown = payload
                    if markdown:
                        ctx["markdown_evidence"] = str(markdown)
                        # Inform stream that markdown evidence was captured
                        yield json.dumps(await self._emit_step(
                            ThinkingPhase.OBSERVATION,
                            json.dumps({
                                "note": "markdown_evidence captured",
                                "length": len(ctx["markdown_evidence"])
                            }),
                            0.9
                        )) + "\n"
            except Exception:
                # Non-fatal: continue workflow even if capture fails
                pass

            # Simplified pipeline: skip LLM observation/reflection.
            # Emit a synthetic observation and proceed directly to final synthesis.
            observation: Dict[str, Any] = {"next_action": "complete", "execution_success": bool(result and result.get("success", True))}
            yield json.dumps(await self._emit_step(ThinkingPhase.OBSERVATION, json.dumps(observation, ensure_ascii=False)[:800])) + "\n"
            break

        # Final synthesis
        if cv_only:
            # Specialized synthesis for CV-only context (streamed)
            yield json.dumps({"type": "status", "message": "Generating CV analysis... this may take a little time"}) + "\n"
            final_parts: List[str] = []
            async for delta in self.llm.stream(
                prompt=json.dumps({
                    "original_query": user_query,
                    "cv_only": True,
                    "tool_results": all_results,
                }, ensure_ascii=False)[:1500],
                system_prompt=ReasoningPrompts.CV_FINAL_SUMMARY,
                temperature=0.2,
                max_tokens=getattr(settings, "LLM_MAX_TOKENS", 800),
            ):
                final_parts.append(delta)
                yield json.dumps({"type": "response", "content": delta}) + "\n"
            final_text = "".join(final_parts)
        else:
            # Provide markdown evidence (truncated) to the LLM as evidence_digest if present
            md = ctx.get("markdown_evidence")
            evidence_digest = (md[:2000] + ("..." if md and len(md) > 2000 else "")) if md else ""
            final_prompt = ReasoningPrompts.FINAL_SYNTHESIS.format(
                original_query=user_query,
                all_results=json.dumps(all_results, ensure_ascii=False)[:1200],
                key_insights="",
                jobs_found="",
                evidence_digest=evidence_digest,
                has_cv=str(has_cv).lower(),
                matching_intent=str(bool(ctx.get("job_url"))).lower(),
            )
            # Stream general final synthesis
            yield json.dumps({"type": "status", "message": "Generating answer... this may take some time depending on content size"}) + "\n"
            final_parts: List[str] = []
            async for delta in self.llm.stream(
                prompt=final_prompt,
                system_prompt=None,
                temperature=0.2,
                max_tokens=getattr(settings, "LLM_MAX_TOKENS", 800),
            ):
                if delta:
                    final_parts.append(delta)
                    yield json.dumps({"type": "response", "content": delta}) + "\n"
            final_text = "".join(final_parts)
        yield json.dumps({"type": "final", "final_response": final_text}) + "\n"