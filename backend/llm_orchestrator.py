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

class RateLimitSettings:
    # Groq rate limits (adjust based on your plan)
    REQUESTS_PER_MINUTE: int = 30
    REQUESTS_PER_HOUR: int = 500
    MAX_BACKOFF: int = 300  # 5 minutes max backoff
    JITTER: bool = True


@dataclass
class RateLimitState:
    requests_made: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    backoff_until: Optional[datetime] = None
    consecutive_failures: int = 0


class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(
        self, 
        requests_per_minute: int = 30,
        requests_per_hour: int = 500,
        max_backoff: int = 300,
        jitter: bool = True
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.max_backoff = max_backoff
        self.jitter = jitter
        
        # Sliding window for minute-level tracking
        self.minute_requests = deque()
        # Hour-level state
        self.hour_state = RateLimitState()
        
        # Adaptive backoff
        self.current_backoff = 1.0
        self.consecutive_success = 0
        
    async def acquire(self) -> None:
        """Wait if necessary to respect rate limits"""
        now = datetime.now()
        
        # Clean old requests from sliding window (older than 1 minute)
        while self.minute_requests and (now - self.minute_requests[0]) > timedelta(minutes=1):
            self.minute_requests.popleft()
        
        # Reset hour window if needed
        if (now - self.hour_state.window_start) > timedelta(hours=1):
            self.hour_state = RateLimitState()
        
        # Calculate wait times
        minute_wait = self._calculate_minute_wait(now)
        hour_wait = self._calculate_hour_wait(now)
        backoff_wait = self._calculate_backoff_wait(now)
        
        total_wait = max(minute_wait, hour_wait, backoff_wait)
        
        if total_wait > 0:
            logger.info(f"Rate limit wait: {total_wait:.2f}s (minute: {minute_wait:.2f}, "
                      f"hour: {hour_wait:.2f}, backoff: {backoff_wait:.2f})")
            await asyncio.sleep(total_wait)
        
        # Record the request
        self.minute_requests.append(now)
        self.hour_state.requests_made += 1
    
    def _calculate_minute_wait(self, now: datetime) -> float:
        """Calculate wait time based on minute rate limit"""
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest = self.minute_requests[0]
            next_available = oldest + timedelta(minutes=1)
            return max(0.0, (next_available - now).total_seconds())
        return 0.0
    
    def _calculate_hour_wait(self, now: datetime) -> float:
        """Calculate wait time based on hourly rate limit"""
        if self.hour_state.requests_made >= self.requests_per_hour:
            next_available = self.hour_state.window_start + timedelta(hours=1)
            return max(0.0, (next_available - now).total_seconds())
        return 0.0
    
    def _calculate_backoff_wait(self, now: datetime) -> float:
        """Calculate wait time based on backoff state"""
        if self.hour_state.backoff_until:
            wait = (self.hour_state.backoff_until - now).total_seconds()
            return max(0.0, wait)
        return 0.0
    
    def on_success(self) -> None:
        """Call when a request succeeds to adjust backoff"""
        self.consecutive_success += 1
        self.current_backoff = max(1.0, self.current_backoff * 0.9)  # Reduce backoff
        
    def on_failure(self, error: Optional[Exception] = None, retry_after_seconds: Optional[float] = None) -> None:
        """Call when a request fails to adjust backoff. If Retry-After is provided, honor it strictly."""
        self.hour_state.consecutive_failures += 1
        self.consecutive_success = 0

        now = datetime.now()
        if retry_after_seconds is not None and retry_after_seconds > 0:
            self.current_backoff = min(self.max_backoff, float(retry_after_seconds))
            self.hour_state.backoff_until = now + timedelta(seconds=self.current_backoff)
            if error:
                logger.warning(f"Request failed (Retry-After {self.current_backoff:.1f}s): {error}")
            return

        # Exponential backoff with jitter when no Retry-After header
        self.current_backoff = min(
            self.max_backoff,
            self.current_backoff * (2 ** min(self.hour_state.consecutive_failures, 5))
        )

        if self.jitter:
            jitter = 1.0 + (random.random() * 0.2 - 0.1)  # ±10% jitter
            self.current_backoff = min(self.max_backoff, self.current_backoff * jitter)

        self.hour_state.backoff_until = now + timedelta(seconds=self.current_backoff)

        if error:
            logger.warning(f"Request failed, backing off for {self.current_backoff:.1f}s: {error}")


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

TOOL DESCRIPTIONS:
1. precheck_job_offer_url: Quick check if a URL is a job page. Use first when URL is provided.
2. fetch_url_html_with_pagination: Fetch HTML from a specific URL. Use when you have a direct URL.
3. html_to_markdown: Convert HTML to markdown. Use after fetching HTML.
4. deep_crawl_job_pages: Crawl multiple job pages. Use only when needed to discover job listings.
5. get_official_website_and_generate_job_urls: Find company website from name. Use as last resort.

RULES:
1. ALWAYS produce at least 1 execution step. Do NOT return an empty plan.
2. If AVAILABLE CONTEXT contains a CV signal (e.g., context.file_path, context.cv_text, context.corrected_text), classify as cv_processing or combined, and include a step to run process_cv_pdf with parameters.file_path = context.file_path.
3. MAX 2-3 STEPS TOTAL. NO LOT OF ITERATIONS: choose the single best path and avoid loops/retries unless strictly necessary.
4. Combine operations where possible (e.g., fetch + markdown in one step). If HTML is already available, proceed to final synthesis immediately (you may pass HTML directly to synthesis if Markdown is not required).
5. Skip discovery phase if URL is provided.
6. Prefer direct URL access over crawling; only crawl if absolutely necessary.
7. ORDERED EXECUTION: precheck (optional) → fetch_html (limit pages) → (optional) html_to_markdown → final synthesis. Stop as soon as sufficient HTML/Markdown is available.

SCENARIOS:
1. DIRECT URL:
   - precheck_url → fetch_html → html_to_markdown
   - Skip crawling unless it's a job board

2. COMPANY NAME ONLY:
   - get_official_website → precheck → fetch_html → html_to_markdown
   - Skip if no clear job section found

3. CV PROCESSING (CV in session):
   - process_cv_pdf with parameters.file_path = context.file_path
   - Only use CV tools
   - No job search tools unless the intent explicitly asks for matching

   - Return helpful response
   - No tool calls unless explicitly requested

4. CV PRESENT IN CONTEXT:
   - Include process_cv_pdf in execution_steps with parameters.file_path = context.file_path

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
            "action": "precheck_job_offer_url",
            "tool_category": "job_search",
            "rationale": "validate page type before fetching",
            "expected_inputs": {{"url": "<from context.job_url>"}},
            "expected_outputs": "Verify if URL is a valid job page",
            "dependencies": []
        }},
        {{
            "step": 2,
            "action": "fetch_url_html_with_pagination",
            "tool_category": "job_search",
            "rationale": "fetch the job page HTML",
            "expected_inputs": {{"url": "<from context.job_url>", "max_pages": 1}},
            "expected_outputs": "Get HTML content of the job page",
            "dependencies": ["1"]
        }}
    ],
    "confidence": 0.95,
    "requires_cv": false
}}

EXAMPLE 2 - CV Analysis (with file in context):
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

Decision Rules and short descriptions (choose only one best fit):
- get_official_website_and_generate_job_urls: Use only when you have a company name but no URL in order to provide a list of urls.
- precheck_job_offer_url: Use when you already have a URL and need to verify it's a job page.
- fetch_url_html_with_pagination: Use when you have a specific URL and need its HTML (possibly across pages).
- deep_crawl_job_pages: Use when starting from a seed URL to find related job pages; avoid if you already have the exact job URL unless broader discovery is required.
- html_to_markdown: Use right after you have HTML content to prepare Markdown for LLM synthesis.


For each available tool, let me evaluate:
1. Does the tool's description match what I need to accomplish?
2. Do the required parameters align with what I have available?
3. Is this the right category (cv_processing vs job_search) for this step?
4. What's the expected input/output format?
5. Are there any dependencies or prerequisites?

I should choose the tool that best matches the step requirements and has the necessary parameters available.

Parameter Requirements (critical):
- Always provide a complete "tool_parameters" object with concrete values (no placeholders).
- If the current step action is "process_cv_pdf" or the step is CV-related and context.file_path exists, set tool_parameters.file_path = context.file_path.
- If context.job_url is present and the tool expects a URL, set:
  - precheck_job_offer_url.tool_parameters.url = context.job_url
  - fetch_url_html_with_pagination.tool_parameters.url = context.job_url
  - deep_crawl_job_pages.tool_parameters.start_url = context.job_url
- For html_to_markdown, set tool_parameters.html_content = "<HTML_FROM_PREVIOUS_STEP>".
- NEVER change or guess domains; use exactly the URL found in the user's message or context.

Minimize iterations:
- Prefer the shortest valid path. If HTML is already present in context (or from a previous step), do NOT refetch/crawl; proceed to final synthesis.
- If Markdown is not necessary, pass HTML evidence directly to synthesis.

When the step is CV-related, prefer process_cv_pdf if context.file_path is available.

Select the best tool from the discovered tools in strictly valid JSON only. Use double quotes for all keys and string values. No comments, no trailing commas, no additional prose or markdown. Output only the JSON object. Ensure tool_parameters are fully specified with concrete values:
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
    "next_action": "continue|retry|pivot|complete",
    "confidence": 0.0-1.0
}}
"""

    REFLECTION = """<thinking>
Let me reflect on the entire process so far.

Initial Query: {initial_query}
Execution Steps Taken: {steps_taken}
Results Obtained: {results}
Current Status: {status}

I should evaluate:
1. Have I addressed the user's original intent?
2. Are the results comprehensive and accurate?
3. Is there additional value I can provide?
4. What could be improved in my approach?
</thinking>

Reflect on the process in strictly valid JSON only. Use double quotes for all keys and string values. No comments, no trailing commas, no additional prose or markdown. Output only the JSON object:
{{
    "query_satisfaction": "fully|partially|not_satisfied",
    "result_completeness": "complete|partial|incomplete", 
    "additional_value": ["what", "else", "could", "help"],
    "process_improvements": ["how", "to", "do", "better"],
    "user_next_steps": ["suggested", "actions", "for", "user"],
    "confidence": 0.0-1.0
}}
"""

    FINAL_SYNTHESIS = """<thinking>
I must deliver a precise, user-centered answer STRICTLY grounded in the extracted evidence. No generic site descriptions. No invented details.

Original Query: {original_query}

Available Evidence (truncated where needed):
- All tool results: {all_results}
- Key insights: {key_insights}
- Tool-extracted jobs (ground truth for listing items): {jobs_found}
- Evidence priority: Markdown if present; otherwise raw HTML is acceptable.
- Evidence digest (top items): {evidence_digest}

Context Flags (MUST honor exactly):
- has_cv: {has_cv}
- matching_intent: {matching_intent}

Decisions:
1) Perform CV-job matching ONLY if has_cv==True AND matching_intent==True. Otherwise, DO NOT compare to a CV.
2) If no jobs/evidence available, transparently explain the limitation and propose the next step.

Output Requirements (STRICT):
1) Be concise and structured.
2) Include URLs only if they appear in evidence (job.url/apply_url/source_url). Never invent or alter URLs.
3) Prefer bullet points. Avoid long prose.
4) Use the page language if obvious from content (French vs English). Default to English otherwise.
5) If any data is uncertain or missing in evidence, state it as “not specified”.

Formatting:
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

End the answer with a short actionable next step (e.g., “Share another company or a specific job URL to go deeper”).
"""

    # System prompt applied when the workflow is recognized as job-related (URL provided or job_search intent)
    JOB_SEARCH_ASSISTANT_SYSTEM = """
You are a Job Search and Extraction Assistant.

Strict rules:
- Only use grounded evidence from fetched/crawled pages and tool outputs.
- Never invent company names, locations, counts, or URLs.
- Prefer concise, structured outputs; avoid long prose.
- If the context includes a direct job URL, prioritize tools in this order:
  1) precheck_job_offer_url (to validate page type)
  2) fetch_url_html_with_pagination (limit 1-3 pages)
  3) html_to_markdown (convert HTML to markdown for extraction)
  4) deep_crawl_job_pages (only if listing page; breadth 5-10 pages, depth <= 2)

Tool parameter mapping rules:
- If context.job_url is present, map it to:
  - precheck_job_offer_url.url = context.job_url
  - fetch_url_html_with_pagination.url = context.job_url
  - deep_crawl_job_pages.start_url = context.job_url
- Always include concrete values; never placeholders.

Observation rules:
- If markdown_evidence already exists, prefer next_action = "complete" to proceed to synthesis.
- If raw HTML is available and sufficient, proceed to synthesis without generating Markdown.
- Only continue crawling when needed (e.g., listing pages) and within safe bounds.

Safety:
- Do not follow external links beyond the configured crawler.
- Do not summarize beyond what is present in markdown evidence or extracted jobs.
"""


    # System prompt applied when CV analysis is requested or a CV file is present in context
    CV_ASSISTANT_SYSTEM = """
You are a CV Analysis Assistant.

Strict rules:
- Plans with empty execution_steps are INVALID. Always produce at least one step.
- If the context includes a CV path (context.file_path) or CV text, you MUST include a step to run process_cv_pdf with expected_inputs.file_path = context.file_path.
- Do not invent paths or placeholders; use concrete values taken from context.
- Keep plans concise (1 step for pure CV analysis).

Observation rules:
- If corrected_text or previous_corrected_text exists in context, prefer next_action = "complete" for synthesis rather than re-processing.

Safety:
- Never hallucinate profile details; only report what tools extracted.
"""


    # Unified workflow system for planning and tool selection
    WORKFLOW_SYSTEM = """
You are a Workflow Manager.

Strict rules:
- Use CV_ASSISTANT_SYSTEM when a CV is present in context.
- Use JOB_SEARCH_ASSISTANT_SYSTEM when a job URL is provided or job_search intent is detected.
- Always include process_cv_pdf in plans when a CV is present.

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


class GroqLLMClient:
    """Groq Chat Completions client with multiple API key rotation."""

    # Model configurations with token limits and capabilities
    # Based on Groq's rate limits: https://console.groq.com/settings/limits
    MODEL_CONFIGS = {
        "llama-3.3-70b-versatile": {
            "max_tokens": 4000,  # Lower max_tokens to fit within rate limits
            "priority": 1,  # Highest quality
            "tokens_per_minute": 12000,
            "requests_per_minute": 30,
            "cost_per_token": 0.0007,  # Relative cost factor
            "best_for": ["complex_analysis", "high_quality_output"]
        },
        "llama-3.1-8b-instant": {
            "max_tokens": 6000,  # Can handle longer contexts
            "priority": 2,  # Balanced option
            "tokens_per_minute": 6000,
            "requests_per_minute": 30,
            "cost_per_token": 0.0004,
            "best_for": ["general_purpose", "code_generation"]
        },
        "gemma2-9b-it": {
            "max_tokens": 8000,  # Best for long contexts
            "priority": 3,  # Highest throughput
            "tokens_per_minute": 15000,
            "requests_per_minute": 30,
            "cost_per_token": 0.0003,
            "best_for": ["long_context", "high_throughput"]
        }
    }

    def _select_best_model(self, prompt_length: int) -> tuple[str, dict]:
        """Select the best model given the prompt length and configured priorities.

        Heuristic:
        - Prefer higher priority models when they have sufficient headroom.
        - Ensure at least 500 tokens headroom beyond prompt_length for response + system.
        - Fall back to the default/self.model if nothing else qualifies.
        """
        required_headroom = 500
        candidates = []
        for name, cfg in self.MODEL_CONFIGS.items():
            max_tokens = cfg.get("max_tokens", 4000)
            if max_tokens - prompt_length > required_headroom:
                candidates.append((cfg.get("priority", 999), name, cfg))

        if candidates:
            candidates.sort(key=lambda x: x[0])  # lower priority value = better
            _, name, cfg = candidates[0]
            return name, cfg

        # Fallback to configured model even if headroom is small
        fallback_name = self.model or next(iter(self.MODEL_CONFIGS))
        return fallback_name, self.MODEL_CONFIGS.get(fallback_name, next(iter(self.MODEL_CONFIGS.values())))

    # Constants for rate limiting
    RATE_LIMIT_RESET_BUFFER = 1.0  # Add 1 second buffer to rate limit reset time
    WINDOW_SIZE = 60  # 1 minute window for rate limiting
    REQUESTS_PER_MINUTE = 60  # Default requests per minute per key
    
    def __init__(self, api_keys: list, model: str = None, timeout: float = 30, loop=None):
        if not isinstance(api_keys, list) or not api_keys:
            raise ValueError("api_keys must be a non-empty list of API keys")
            
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model = model or next(iter(self.MODEL_CONFIGS))  # Default to first model
        self.timeout = float(timeout)
        self.base_url = settings.GROQ_API_URL
        # Normalize base URL: remove trailing slash and accidental endpoint suffix
        if isinstance(self.base_url, str):
            self.base_url = self.base_url.rstrip('/')
            if self.base_url.endswith('/chat/completions'):
                self.base_url = self.base_url[: -len('/chat/completions')]
        self._last_call_time = 0
        # Gentle throttling: increase min interval; allow override via settings
        self._min_call_interval = float(getattr(settings, 'LLM_MIN_CALL_INTERVAL', 2.0))
        self._loop = loop or asyncio.get_event_loop()
        # Global lock to serialize outbound Groq calls and avoid spikes
        self._call_lock = asyncio.Lock()
        
        # Initialize rate limiters for each key
        self._rate_limiters = {
            key: RateLimiter(
                requests_per_minute=RateLimitSettings.REQUESTS_PER_MINUTE,
                requests_per_hour=RateLimitSettings.REQUESTS_PER_HOUR,
                max_backoff=RateLimitSettings.MAX_BACKOFF,
                jitter=RateLimitSettings.JITTER
            ) for key in api_keys
        }
        
        # Initialize key usage tracking
        self.key_usage = {
            key: {
                "requests_in_window": 0,
                "requests": 0,
                "last_used": 0,
                "window_start": time.time(),
                "model_usage": {
                    model_name: {"tokens_used": 0} 
                    for model_name in self.MODEL_CONFIGS
                },
                "rate_limited_until": 0
            } for key in api_keys
        }
        
        # Initialize HTTP clients
        self._clients = {}
        self._initialize_clients()
        
    async def _check_groq_health(self):
        """Check if Groq API is accessible and model is available."""
        test_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0
        }
        
        # Try each key until one works
        for key, client in self._clients.items():
            rate_limiter = self._rate_limiters[key]
            
            try:
                # Respect rate limits
                await rate_limiter.acquire()
                
                # Make the request
                async with async_timeout.timeout(10):  # Shorter timeout for health check
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=test_payload
                    )
                    
                response.raise_for_status()
                rate_limiter.on_success()
                logger.info(f"✅ Groq API is accessible with model: {self.model}")
                return True, None
                
            except httpx.HTTPStatusError as e:
                rate_limiter.on_failure(e)
                if e.response.status_code == 401:
                    logger.error(f"Groq API health check failed: Invalid API key ending in {key[-4:]}")
                    return False, f"Invalid Groq API key (ending in {key[-4]})"
                    
                logger.error(f"Groq API health check failed with HTTP {e.response.status_code}: {e.response.text}")
                return False, f"Groq API error: HTTP {e.response.status_code}"
                
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                rate_limiter.on_failure(e)
                logger.error(f"Groq API health check failed due to a request error: {type(e).__name__} - {e}")
                # Don't fail on network errors, try the next key
                continue
                
        return False, "No working API keys available"

    def _initialize_clients(self):
        """Initialize HTTP clients for each API key."""
        self._clients = {}
        for key in self.api_keys:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            self._clients[key] = httpx.AsyncClient(timeout=self.timeout, headers=headers)
    
    def _get_next_key(self, last_key_status: Optional[Dict] = None):
        """Get the next available API key with smart rate limiting."""
        current_time = time.time()
        
        # Handle rate limiting for the last used key
        if last_key_status and last_key_status.get('rate_limited', False):
            key = last_key_status['key']
            block_time = 120  # 2 minutes block for rate limited keys
            self.key_usage[key]['rate_limited_until'] = current_time + block_time
            logger.warning(f"Key {key[-4:]} rate limited. Blocking for {block_time}s")
        
        # Reset request counts if window has passed
        for key in self.api_keys:
            if current_time - self.key_usage[key]['window_start'] > self.WINDOW_SIZE:
                self.key_usage[key]['requests_in_window'] = 0
                self.key_usage[key]['window_start'] = current_time
        
        # Find next available key
        best_key = None
        min_requests = float('inf')
        
        now_dt = datetime.now()
        for key in self.api_keys:
            key_data = self.key_usage[key]
            # Also respect per-key backoff from RateLimiter
            rl = self._rate_limiters.get(key)
            if rl and rl.hour_state.backoff_until and rl.hour_state.backoff_until > now_dt:
                continue
            
            # Skip if rate limited
            if key_data['rate_limited_until'] > current_time:
                continue
                
            # Find the least used key in current window
            if key_data['requests_in_window'] < min_requests:
                min_requests = key_data['requests_in_window']
                best_key = key
        
        if best_key:
            # Enforce rate limiting
            if self.key_usage[best_key]['requests_in_window'] >= self.REQUESTS_PER_MINUTE:
                wait_time = (self.key_usage[best_key]['window_start'] + self.WINDOW_SIZE) - current_time
                if wait_time > 0:
                    logger.warning(f"Rate limit reached for key {best_key[-4:]}. Waiting {wait_time:.1f}s")
                    time.sleep(wait_time + 0.1)  # Small buffer
                    return self._get_next_key()  # Retry after waiting
            
            # Update key usage
            self.key_usage[best_key]['requests'] += 1
            self.key_usage[best_key]['requests_in_window'] += 1
            self.key_usage[best_key]['last_used'] = current_time
            
            logger.info(f"Using key {best_key[-4:]} (used {self.key_usage[best_key]['requests_in_window']}/{self.REQUESTS_PER_MINUTE} this minute)")
            return best_key, self._clients[best_key], self.key_usage[best_key]
        
        # If all keys are rate limited, wait for the first one to become available
        wait_time = min(
            (self.key_usage[key]['rate_limited_until'] - current_time 
             for key in self.api_keys),
            default=120  # Default 2 minutes if no keys are rate limited (shouldn't happen)
        )
        wait_time = max(min(wait_time, 120), 10)  # Cap between 10 and 120 seconds
        logger.warning(f"All keys rate limited. Waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        return self._get_next_key()  # Retry after waiting
        # Minimal client: no concurrency semaphore, cooldowns, or inter-call delays

    async def _enforce_rate_limit(self):
        """Ensure minimum time between API calls"""
        now = time.time()
        time_since_last_call = now - getattr(self, '_last_api_call_time', 0)
        min_interval = 1.0  # Minimum 1 second between calls
        
        if time_since_last_call < min_interval:
            wait_time = min_interval - time_since_last_call
            logger.debug(f"Rate limiting: Waiting {wait_time:.2f}s between API calls")
            await asyncio.sleep(wait_time)
        
        self._last_api_call_time = time.time()

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 5,
        last_error: Optional[Exception] = None,
        max_delay: float = 60.0,  # Maximum backoff delay in seconds
        enforce_rate_limit: bool = True  # Allow disabling rate limiting when needed
    ) -> str:
        """Generate text using Groq API with advanced rate limiting and automatic retry."""
        # Select or determine the best model to use
        prompt_length = len(prompt.split())  # Rough estimate
        selected_model, model_config = self._select_best_model(prompt_length) if model is None else \
                                     (model, self.MODEL_CONFIGS.get(model, next(iter(self.MODEL_CONFIGS.values()))))
        
        # Calculate max_tokens if not provided
        if max_tokens is None:
            max_tokens = min(
                model_config["max_tokens"] - prompt_length - 100,  # Leave room for response
                4000  # Absolute max
            )
            max_tokens = max(max_tokens, 100)  # Ensure minimum
        
        # Prepare the request payload
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "stream": False,
        }
        
        # Track attempts and errors
        last_error = None
        attempt = 0
        max_attempts = len(self.api_keys) * 2  # Try each key at least twice
        
        while attempt < max_attempts:
            # Get the next available key and its rate limiter
            key, client, _ = self._get_next_key()
            rate_limiter = self._rate_limiters[key]
            key_suffix = key[-4:]
            
            try:
                # Serialize calls to avoid bursts across tasks
                async with self._call_lock:
                    # Respect rate limits before making the request
                    await rate_limiter.acquire()
                    
                    # Enforce minimum delay between API calls
                    now = time.time()
                    time_since_last_call = now - self._last_call_time
                    if time_since_last_call < self._min_call_interval:
                        wait_time = self._min_call_interval - time_since_last_call
                        logger.debug(f"Enforcing rate limit: waiting {wait_time:.2f}s between API calls")
                        await asyncio.sleep(wait_time)
                    
                    # Make the API request with timeout
                    self._last_call_time = time.time()
                    logger.info(f"Sending request to {selected_model} using key ending in {key_suffix} (Attempt {attempt + 1}/{max_attempts})")
                    
                    async with async_timeout.timeout(self.timeout):
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            json=payload
                        )
                
                # Check for rate limit headers
                headers = response.headers
                if 'x-ratelimit-remaining' in headers and 'x-ratelimit-reset' in headers:
                    remaining = int(headers['x-ratelimit-remaining'])
                    reset_time = float(headers['x-ratelimit-reset'])
                    logger.debug(f"Rate limit: {remaining} requests remaining, reset in {reset_time}s")
                
                response.raise_for_status()
                
                # Update rate limiter on success
                rate_limiter.on_success()
                
                # Process successful response
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    raise ValueError("Unexpected response format from API")
                    
            except httpx.HTTPStatusError as e:
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get('retry-after')) if e.response is not None else None
                except Exception:
                    retry_after = None
                rate_limiter.on_failure(e, retry_after_seconds=retry_after)
                last_error = e
                
                if e.response.status_code == 429:  # Rate limited
                    # Use server value, but cap our cooldown to avoid excessively long sleeps; we won't reuse this key until cooldown ends anyway
                    if retry_after is None:
                        retry_after = 1.0
                    max_retry_after = float(getattr(settings, 'LLM_MAX_RETRY_AFTER', 120.0))
                    capped = min(float(retry_after), max_retry_after)
                    logger.warning(f"Rate limited on key {key_suffix}. Retry-After={retry_after}s (capped cooldown to {capped}s)")
                    # Mark key cooldown in key_usage as well
                    self.key_usage[key]['rate_limited_until'] = time.time() + capped
                    # Do NOT sleep here; try another key immediately
                    attempt += 1
                    # Before continuing, if all keys are cooling down, sleep until the nearest cooldown expires
                    now_ts = time.time()
                    next_available_in = []
                    now_dt_inner = datetime.now()
                    for k in self.api_keys:
                        ku = self.key_usage[k]
                        rlk = self._rate_limiters.get(k)
                        blocked_by_header = max(0.0, ku['rate_limited_until'] - now_ts)
                        blocked_by_rl = 0.0
                        if rlk and rlk.hour_state.backoff_until and rlk.hour_state.backoff_until > now_dt_inner:
                            blocked_by_rl = (rlk.hour_state.backoff_until - now_dt_inner).total_seconds()
                        next_available_in.append(max(blocked_by_header, blocked_by_rl))
                    min_wait = min(next_available_in) if next_available_in else 0.0
                    if min_wait > 0:
                        sleep_for = min(min_wait, max_retry_after)
                        logger.info(f"All keys cooling down. Waiting {sleep_for:.1f}s until next key is available")
                        await asyncio.sleep(sleep_for)
                    continue
                else:
                    logger.error(f"HTTP error {e.response.status_code} from key {key_suffix}: {e.response.text}")
                    attempt += 1
                    
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                rate_limiter.on_failure(e)
                last_error = e
                logger.error(f"Request failed on key {key_suffix}: {str(e)}")
                attempt += 1
                
            except Exception as e:
                rate_limiter.on_failure(e)
                last_error = e
                logger.error(f"Unexpected error on key {key_suffix}: {str(e)}")
                attempt += 1
        
        # If we get here, all attempts failed
        error_msg = f"Failed after {max_attempts} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        raise RuntimeError(error_msg)
        
        # If we've exhausted all retries, raise the last error
        if last_err:
            raise last_err
        raise Exception("Failed to complete request after maximum retries")
                
    async def _should_use_llm(self, prompt: str) -> bool:
        """
        Analyze the input to determine if an LLM call is necessary.
        Returns True if LLM processing is needed, False otherwise.
        """
        # Simple heuristics to avoid unnecessary LLM calls
        simple_queries = {
            "hello", "hi", "thanks", "thank you", "ok", "okay",
            "yes", "no", "maybe", "help", "?"
        }

        # Check if prompt is empty or very short
        if not prompt or len(prompt.strip()) < 3:
            return False

        prompt_lower = prompt.lower().strip()

        # Whole-word match for very short messages only (<= 3 words)
        words = re.findall(r"\b\w+\b", prompt_lower)
        if 0 < len(words) <= 3:
            for w in words:
                if w in simple_queries:
                    return False
        # Also handle exact short punctuation-only queries
        if prompt_lower in simple_queries:
            return False

        # Check if it's a simple command
        if prompt_lower.startswith(('/help', '/start', '/info', '/about')):
            return False

        return True
        
    async def generate_with_reasoning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        temperature: Optional[float] = 0.7,
        force: bool = False
    ) -> str:
        """
        Generate text with reasoning to minimize LLM calls.
        First determines if LLM call is needed, then processes accordingly.
        """
        # First, check if we can avoid LLM call (allow bypass via force)
        needs_llm = True if force else await self._should_use_llm(prompt)
        if not needs_llm:
            logger.info("Skipping LLM call - simple query detected")
            return "I'm here to help! Could you please provide more details about what you're looking for?"
            
        # If we get here, we'll use the LLM but with rate limiting
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            model=model,
            max_retries=max_retries
        )
        
    async def aclose(self):
        # Close all underlying HTTP clients
        try:
            if hasattr(self, '_clients') and isinstance(self._clients, dict):
                for key, client in list(self._clients.items()):
                    try:
                        await client.aclose()
                    except Exception as e:
                        logger.warning(f"Error closing HTTP client for key {str(key)[-4:]}: {e}")
        except Exception as e:
            logger.warning(f"Error during GroqLLMClient shutdown: {e}")


class LLMOrchestrator:
    """LLM-first orchestrator with streaming of thinking steps."""

    def __init__(self):
        # Get API keys from environment, support both single key and comma-separated list
        api_keys = [key.strip() for key in settings.GROQ_API_KEY.split(',') if key.strip()]
        
        # Prefer GROQ_MODEL if provided; fall back to LLM_MODEL for compatibility
        selected_model = getattr(settings, 'GROQ_MODEL', None) or getattr(settings, 'LLM_MODEL', None)
        self.llm = GroqLLMClient(
            api_keys=api_keys,
            model=selected_model,
            timeout=settings.GROQ_TIMEOUT,
        )
        logger.info(f"Initialized Groq client with {len(api_keys)} API keys")
        self.cv_client: Optional[CVClient] = None
        self.job_client: Optional[JobCrawlerClient] = None
        self.thinking_history: List[ThinkingStep] = []
        self.context: Dict[str, Any] = {}

    async def initialize(self):
        logger.info("Initializing LLM Orchestrator (Groq)…")
        # Init MCP clients (timeout-safe), tolerate partial availability
        # CV MCP
        try:
            self.cv_client = CVClient(settings.CV_MCP_SERVER_SCRIPT)
            await self.cv_client.start()
            logger.info("CV MCP connected and tools discovered")
        except Exception as e:
            logger.warning(f"CV MCP unavailable, continuing without CV tools: {e}")
            self.cv_client = None

        # Job MCP (optional)
        try:
            script = (settings.JOB_MCP_SERVER_SCRIPT or '').strip()
            if not script or not Path(script).exists():
                logger.warning("Job MCP server script not provided or not found; skipping job tools")
                self.job_client = None
            else:
                self.job_client = JobCrawlerClient(script)
                await self.job_client.start()
                logger.info("Job MCP connected and tools discovered")
        except Exception as e:
            logger.warning(f"Job MCP unavailable, continuing without job tools: {e}")
            self.job_client = None

        logger.info(
            "LLM Orchestrator ready (cv_tools=%s, job_tools=%s)",
            bool(self.cv_client), bool(self.job_client)
        )

    async def shutdown(self):
        if self.cv_client:
            await self.cv_client.stop()
        if self.job_client:
            await self.job_client.stop()
        if self.llm:
            await self.llm.aclose()

    async def _extract_jobs_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Extract jobs as structured JSON from markdown using the existing LLM client.

        Args:
            markdown_content: Raw markdown content containing job listings
            
        Returns:
            List of dictionaries containing structured job information
            
        Raises:
            ValueError: If markdown content is empty or invalid
            RuntimeError: If LLM response cannot be parsed
        """
        if not markdown_content or not markdown_content.strip():
            logger.warning("Empty markdown content provided")
            return []
            
        try:
            prompt = (
                "You are a precise information extraction engine. Input may include MULTIPLE sources. Each source is delimited as:\n"
                "### SOURCE_BEGIN <URL>\n"
                "<markdown content>\n"
                "### SOURCE_END\n\n"
                "Task: extract ALL distinct job listings across ALL sources into a compact JSON array. Use keys: \n"
                "'title', 'position_name', 'location', 'sector', 'description', 'company', 'url', 'apply_url', 'requirements_snippet', 'salary', 'contract_type', 'required_skill', 'post_date', 'source_url', 'experience', 'mobility', 'languages'. "
                "Also accept French labels and map them to these keys where applicable: 'expérience'->experience, 'rémunération' or 'rémunération proposée'->salary, 'mobilité'->mobility, 'langues'->languages, 'Job Posting Description'->description. "
                "For textual fields (title, position_name, location, sector, description, company, requirements_snippet, salary, contract_type, required_skill, post_date, experience, mobility, languages), if missing, set to the literal 'not mentioned'. "
                "For 'url' and 'apply_url', do NOT fabricate links; include only if explicitly present in the content. \n"
                "For 'source_url', ALWAYS set it to the URL from the nearest enclosing SOURCE_BEGIN header for that job.\n"
                "If there is a section like 'Profil', 'Compétences', 'Exigences', 'Requirements', include 1-3 concise bullet points or a short paragraph as 'requirements_snippet'. "
                "Deduplicate identical jobs. Return ONLY the JSON array, no prose.\n\n"
                f"INPUT (multi-source markdown):\n{markdown_content[:12000]}"
            )
            
            # Generate response with timeout and error handling
            try:
                response = await asyncio.wait_for(
                    self.llm.generate(prompt, system_prompt=None, max_tokens=800),
                    timeout=settings.MCP_TOOL_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error("LLM generation timed out")
                raise RuntimeError("Job extraction timed out") from None
                
            if not response:
                logger.error("Empty response from LLM")
                return []
                
            # Parse the JSON response
            try:
                data = await self._parse_json(response)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                return []
                
            if not isinstance(data, list):
                logger.error(f"Expected list of jobs, got {type(data).__name__}")
                return []
                
            # Normalize and validate job entries
            norm_jobs: List[Dict[str, Any]] = []
            for job in data:
                if not isinstance(job, dict):
                    continue
                    
                out: Dict[str, Any] = {}
                
                # Extract and normalize fields
                fields = [
                    ('title', ['title', 'job_title', 'position', 'position_name']),
                    ('company', ['company', 'employer']),
                    ('location', ['location', 'city']),
                    ('sector', ['sector', 'secteur', 'category']),
                    ('description', ['description', 'summary', 'details']),
                    ('requirements_snippet', ['requirements_snippet', 'requirements', 'profil', 'exigences']),
                    ('experience', ['experience', 'expérience', 'experiences', 'years_of_experience', 'experience_level']),
                    ('mobility', ['mobility', 'mobilite', 'mobilité', 'travel', 'déplacement', 'deplacement'])
                ]
                
                for target_key, source_keys in fields:
                    for src_key in source_keys:
                        if src_key in job and job[src_key]:
                            out[target_key] = job[src_key]
                            break
                    
                # Special handling for position_name
                if 'title' in out and 'position_name' not in out:
                    out['position_name'] = out['title']
                    
                # Add URL if present
                for url_key in ['url', 'apply_url', 'source_url']:
                    if url_key in job and job[url_key]:
                        out[url_key] = job[url_key]
                
                # Handle language field
                lang_val = job.get('languages') or job.get('langues') or job.get('language') or job.get('langage')
                if lang_val:
                    if isinstance(lang_val, list):
                        try:
                            out['languages'] = ", ".join([str(x) for x in lang_val if str(x).strip()])
                        except Exception:
                            out['languages'] = str(lang_val)
                    else:
                        out['languages'] = str(lang_val)
                
                # Handle job posting description fallback
                jpd = job.get('job_posting_description') or job.get('posting_description')
                if jpd and 'description' not in out:
                    out['description'] = jpd
                    
                # Handle salary fields (including French variants)
                sal_alt = job.get('rémunération') or job.get('remuneration') or \
                         job.get('rémunération_proposée') or job.get('remuneration_proposee') or \
                         job.get('salary')
                if sal_alt and 'salary' not in out:
                    out['salary'] = sal_alt
                    
                # Handle contract type
                contract = job.get('contract_type') or job.get('type_contrat') or job.get('contract')
                if contract:
                    out['contract_type'] = contract
                    
                # Handle required skills
                if 'required_skill' not in out:
                    skills = job.get('required_skill') or job.get('compétences_requises') or \
                            job.get('skills') or job.get('compétences')
                    if skills:
                        out['required_skill'] = skills
                
                # Handle URLs
                for key in ['apply_url', 'url', 'link']:
                    if job.get(key) and key not in out:
                        out[key] = job[key]
                
                # Handle source URL
                if job.get('source_url') and 'source_url' not in out:
                    out['source_url'] = job['source_url']
                
                # Set default values for required fields
                for field in ['title', 'company', 'location', 'sector', 'description']:
                    if field not in out:
                        out[field] = 'not mentioned'
                
                norm_jobs.append(out)
                
            return norm_jobs
            
        except Exception as e:
            logger.error(f"Error extracting jobs from markdown: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract jobs: {e}") from e

    async def _emit_step(self, phase: ThinkingPhase, content: str, confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Emit a thinking step with structured metadata and improved display formatting.
        
        Args:
            phase: The current thinking phase
            content: The content to display (can be JSON string or plain text)
            confidence: Confidence score (0.0-1.0)
            metadata: Additional metadata for the step
            
        Returns:
            Dict containing the formatted step information
        """
        if not isinstance(content, str):
            content = str(content)
            
        # Clean up the content for display
        display_content = content.strip()
        
        # If content looks like JSON, try to parse it for better display
        if display_content.startswith('{') and display_content.endswith('}'):
            try:
                parsed = json.loads(display_content)
                if isinstance(parsed, dict):
                    # Format common fields in a more readable way
                    if "action" in parsed and "tool" in parsed:
                        display_content = f"Action: {parsed.get('action', 'N/A')}\nTool: {parsed.get('tool', 'N/A')}"
                        if "parameters" in parsed and parsed["parameters"]:
                            display_content += f"\nParameters: {json.dumps(parsed['parameters'], indent=2, ensure_ascii=False)}"
                        if "rationale" in parsed:
                            display_content += f"\nRationale: {parsed['rationale']}"
                    elif "error" in parsed:
                        display_content = f"Error: {parsed.get('error', 'Unknown error')}"
                        if "suggestion" in parsed:
                            display_content += f"\nSuggestion: {parsed['suggestion']}"
                        if "raw" in parsed:
                            raw_preview = str(parsed['raw'])[:100]
                            if len(str(parsed['raw'])) > 100:
                                raw_preview += "..."
                            display_content += f"\nRaw preview: {raw_preview}"
                    elif "status" in parsed and "message" in parsed:
                        display_content = f"{parsed['status']}: {parsed['message']}"
                    else:
                        # For other JSON, pretty print it
                        display_content = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON in _emit_step: {e}")
                # Try to extract a meaningful message from the content
                error_match = re.search(r'(?i)(error|exception|failed|invalid|missing|required)[\s:]+(.*?)(?:\.|\n|$)', display_content)
                if error_match:
                    error_type = error_match.group(1).capitalize()
                    error_msg = error_match.group(2).strip()
                    display_content = f"{error_type}: {error_msg}"
                    
                    # Try to find line numbers or specific fields mentioned in the error
                    line_match = re.search(r'(?i)line\s+(\d+)', display_content)
                    if line_match:
                        display_content += f"\nError at line {line_match.group(1)}"
                        
                    field_match = re.search(r'(?i)field\s+["\']([^"\']+)["\']', display_content)
                    if field_match:
                        display_content += f"\nField with issue: {field_match.group(1)}"
                else:
                    # If we can't extract a specific error, show the first 200 chars
                    display_content = f"Error parsing response: {display_content[:200]}"
                    if len(content) > 200:
                        display_content += "..."
            except Exception as e:
                logger.warning(f"Unexpected error in _emit_step: {e}", exc_info=True)
                display_content = f"Error: {str(e)}\n\nRaw content: {content[:300]}"
                if len(content) > 300:
                    display_content += "..."
        
        # Create the step with cleaned content
        step = ThinkingStep(
            phase=phase,
            content=content,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.thinking_history.append(step)
        
        # Return the structured data
        return {
            "type": "thinking_step",
            "phase": phase.value,
            "content": display_content,  # Use the cleaned/parsed content
            "confidence": confidence,
            "timestamp": step.timestamp.isoformat(),
            "metadata": metadata or {}
        }

    async def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text with robust error handling and recovery.
        
        Args:
            text: Input text that may contain JSON
            
        Returns:
            Parsed JSON as dict or dict with error information
        """
        if not text or not isinstance(text, str):
            return {"error": "Empty or invalid input", "raw": str(text)[:500]}
            
        # Clean up the text first
        text = text.strip()
        
        # Try to extract JSON from markdown code block first
        if "```json" in text:
            try:
                s = text.find("```json") + 7
                e = text.find("```", s)
                if e > s:  # Make sure we found the closing ```
                    segment = text[s:e].strip()
                    if segment:  # Don't try to parse empty segments
                        return json.loads(segment)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown: {e}")
                # Continue to try other methods
                pass
            except Exception as e:
                logger.warning(f"Unexpected error parsing markdown JSON: {e}")
                # Continue to try other methods
                pass
        
        # Try to find and extract a JSON object
        try:
            # Look for the first { and try to parse until the matching }
            start = text.find('{')
            if start >= 0:
                # Find the matching closing brace
                brace_count = 1
                end = start + 1
                in_string = False
                escape = False
                
                while end < len(text) and brace_count > 0:
                    if not in_string:
                        if text[end] == '{' and not escape:
                            brace_count += 1
                        elif text[end] == '}' and not escape:
                            brace_count -= 1
                        elif text[end] == '"' and not escape:
                            in_string = True
                    else:  # in string
                        if text[end] == '"' and not escape:
                            in_string = False
                        elif text[end] == '\\' and not escape:
                            escape = True
                        else:
                            escape = False
                    end += 1
                
                if brace_count == 0:  # Found balanced braces
                    json_str = text[start:end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to recover by cleaning up common issues
                        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Trailing commas
                        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)  # Unquoted keys
                        return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON with brace matching: {e}")
        
        # As a last resort, try to find any JSON-like object using regex
        try:
            # Look for the most complete JSON object we can find
            json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"Regex-based JSON extraction failed: {e}")
        
        # If all else fails, try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Final JSON parse failed: {e}")
            # Try to extract error location information
            error_msg = str(e)
            if "line" in error_msg and "column" in error_msg:
                line_match = re.search(r'line (\d+) column (\d+)', error_msg)
                if line_match:
                    line_num = int(line_match.group(1))
                    col_num = int(line_match.group(2))
                    lines = text.splitlines()
                    if line_num <= len(lines):
                        error_line = lines[line_num - 1]
                        error_msg += f"\nNear: {error_line[:col_num]}👉{error_line[col_num:col_num+10]}..."
            # Last-chance recovery: sanitize and try Python literal eval (handles single quotes, True/False/None)
            try:
                import ast
                cleaned = text.strip()
                # Strip generic code fences ```...```
                if cleaned.startswith("```") and cleaned.endswith("```"):
                    cleaned = cleaned.strip("`")
                # Normalize smart quotes
                cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
                # Attempt literal eval
                obj = ast.literal_eval(cleaned)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list):
                    # Wrap list to keep a dict shape for downstream validators
                    return {"items": obj}
            except Exception:
                pass

        # Return error information with context
        return {
            "error": "Failed to parse JSON response",
            "details": str(e) if 'e' in locals() else "Unknown error",
            "snippet": text[:500] + ("..." if len(text) > 500 else ""),
            "suggestion": "The LLM response was not in a valid JSON format. Please check the prompt and model output."
        }

    def _validate_schema(self, model_cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against model_cls. Return data with optional validation_errors array."""
        try:
            # Allow partial dicts: construct then dump, preserving known fields
            model = model_cls.model_validate(data)
            return model.model_dump()
        except ValidationError as ve:
            # Attach errors but keep original data to avoid breaking the flow
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
        if self.cv_client:
            try:
                await self.cv_client.discover_tools()
                for t in self.cv_client.tools:
                    discovered["cv_processing"]["tools"].append({
                        "name": t.name,
                        "description": getattr(t, "description", "") or getattr(t, "title", ""),
                        "schema": getattr(t, "inputSchema", {}),
                    })
            except Exception as e:
                logger.error(f"Discover CV tools failed: {e}")
        if self.job_client:
            try:
                await self.job_client.discover_tools()
                for t in self.job_client.tools:
                    discovered["job_search"]["tools"].append({
                        "name": t.name,
                        "description": getattr(t, "description", "") or getattr(t, "title", ""),
                        "schema": getattr(t, "inputSchema", {}),
                    })
            except Exception as e:
                logger.error(f"Discover Job tools failed: {e}")
        return discovered

    async def process_query_stream(self, query: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        start = time.time()
        self.thinking_history = []
        if context:
            self.context.update(context)

        # Heuristic: extract a URL from the user's query for better routing
        try:
            import re
            url_match = re.search(r"https?://[^\s\"]+", query or "")
            if url_match and not self.context.get("job_url"):
                self.context["job_url"] = url_match.group(0)
                # Emit a status update about the detected URL
                yield json.dumps({
                    "type": "status",
                    "message": f"Detected URL: {url_match.group(0)}",
                    "phase": "initial_analysis"
                }) + "\n"
        except Exception as e:
            logger.warning(f"Error extracting URL from query: {e}")
            

        # Helpers to keep prompts small (avoid 413)
        def _compact_context_for_prompt() -> Dict[str, Any]:
            try:
                ctx = dict(self.context)
                # Drop large fields
                ctx.pop("markdown_evidence", None)
                # Truncate long strings
                for k, v in list(ctx.items()):
                    if isinstance(v, str) and len(v) > 500:
                        ctx[k] = v[:500] + "…"
                return ctx
            except Exception:
                return {}

        def _summarize_tools_for_prompt(tools_map: Dict[str, Any], max_per_cat: int = 6, max_desc: int = 160) -> Dict[str, Any]:
            small = {}
            try:
                for category, cat_data in tools_map.items():
                    tools = cat_data.get("tools", [])[:max_per_cat]
                    small[category] = {
                        "description": cat_data.get("description", ""),
                        "tools": [
                            {
                                "name": t.get("name", ""),
                                "description": (t.get("description", "") or "")[:max_desc]
                            }
                            for t in tools
                        ]
                    }
            except Exception as e:
                logger.warning(f"Error summarizing tools: {e}")
            return small

        # Phase 1: Analysis
        analysis_prompt = ReasoningPrompts.MERGED_ANALYSIS_AND_PLAN.format(
            query=query or "",
            context=json.dumps(_compact_context_for_prompt(), ensure_ascii=False)[:3000]
        )
        
        # Emit analysis thinking step
        try:
            yield json.dumps(await self._emit_step(
                ThinkingPhase.INITIAL_ANALYSIS, 
                "Analyzing query and available context...", 
                0.3,
                {"phase_start": True}
            )) + "\n"
        except Exception as e:
            logger.warning(f"Error emitting analysis step: {e}")
            
        # Get initial analysis
        analysis_text = await self.llm.generate(analysis_prompt, max_tokens=500)
        analysis = await self._parse_json(analysis_text)
        if isinstance(analysis, dict) and not analysis.get("error"):
            analysis = self._validate_schema(AnalysisSchema, analysis)
        
        # Store analysis in context
        self.context["analysis"] = analysis
        
        # Phase 2: Planning (with dynamic discovery)
        # Decide whether to apply a system prompt (job or CV) for subsequent phases
        workflow_system = None
        try:
            qtype = (analysis.get("query_type") or "").lower() if isinstance(analysis, dict) else ""
            if "job" in qtype or qtype in ("job_search", "combined"):
                workflow_system = ReasoningPrompts.JOB_SEARCH_ASSISTANT_SYSTEM
            else:
                # heuristic: treat as job workflow if URL present in query/context
                if ("http" in (query or "")) or (isinstance(self.context, dict) and any(k in self.context for k in ["job_url", "urls", "possible_job_urls"])):
                    workflow_system = ReasoningPrompts.JOB_SEARCH_ASSISTANT_SYSTEM
                # heuristic: treat as CV workflow if CV signals present
                elif isinstance(self.context, dict) and (self.context.get("file_path") or self.context.get("cv_text") or self.context.get("corrected_text") or self.context.get("previous_corrected_text")):
                    workflow_system = ReasoningPrompts.CV_ASSISTANT_SYSTEM
        except Exception as e:
            logger.warning(f"Error determining job system prompt: {e}")

        tools = await self._discover_all_tools()
        # If job-related request but no job tools discovered, inform the user early
        try:
            job_tools = [t.get("name") for t in tools.get("job_search", {}).get("tools", [])]
            is_job_like = ("http" in (query or "")) or (isinstance(self.context, dict) and (self.context.get("job_url") or self.context.get("urls")))
            if is_job_like and not job_tools:
                warn_msg = {
                    "note": "Job tools unavailable",
                    "action_required": "Configure settings.JOB_MCP_SERVER_SCRIPT to a valid Job MCP server path and restart.",
                    "detected_url": self.context.get("job_url") or None
                }
                logger.debug("STREAM OUT thinking_step: planning (job tools unavailable)")
                yield json.dumps(await self._emit_step(ThinkingPhase.PLANNING, "Job tools not available; cannot proceed with crawling.", 0.2, warn_msg)) + "\n"
        except Exception:
            pass
        plan_prompt = ReasoningPrompts.PLANNING.format(
            analysis=json.dumps(analysis, ensure_ascii=False)[:3000],
            available_tools=json.dumps(_summarize_tools_for_prompt(tools), ensure_ascii=False)[:3000],
        )
        try:
            # Heartbeat before long LLM call
            try:
                logger.debug("STREAM HB status: starting planning LLM call")
                yield json.dumps({"type": "status", "phase": "planning", "message": "Creating plan..."}) + "\n"
            except Exception:
                pass
            # Emit a phase-start thinking step for planning
            try:
                yield json.dumps(await self._emit_step(ThinkingPhase.PLANNING, "Drafting plan based on analysis and tools...", 0.5, {"phase_start": True})) + "\n"
            except Exception:
                pass
            # Planning: moderately sized output
            plan_text = await self.llm.generate(plan_prompt, system_prompt=workflow_system, max_tokens=700)
            plan = await self._parse_json(plan_text)
            if isinstance(plan, dict) and not plan.get("error"):
                plan = self._validate_schema(PlanSchema, plan)
            plan["discovered_tools"] = tools
        except Exception as e:
            plan = {"error": str(e), "execution_steps": []}
        try:
            # Stream full plan details as content (JSON string)
            plan_content = json.dumps(plan, ensure_ascii=False)
        except Exception:
            plan_content = "{}"
        logger.debug("STREAM OUT thinking_step: planning")
        yield json.dumps(await self._emit_step(ThinkingPhase.PLANNING, plan_content, float(plan.get("confidence", 0.6)), plan)) + "\n"

        # Enforce: Do not proceed without a functional plan
        try:
            steps_list = plan.get("execution_steps") if isinstance(plan, dict) else []
        except Exception:
            steps_list = []
        if not steps_list:
            # Inform the stream and stop without emitting a final response
            try:
                yield json.dumps({
                    "type": "error",
                    "phase": "planning",
                    "message": "Planning returned no execution steps; cannot proceed to execution."
                }) + "\n"
            except Exception:
                pass
            return

        all_results: List[Dict[str, Any]] = []
        # bucket to aggregate markdown evidence from job pages
        self.context.setdefault("markdown_evidence", [])

        # Phase 3-5: For each planned step, select tool -> execute -> observe
        aborted_plan = False
        executed_tools_count = 0
        for step in plan.get("execution_steps", []):
            # Tool selection
            tool_descs: List[Dict[str, Any]] = []
            for cat, info in tools.items():
                for t in info["tools"]:
                    tool_descs.append({
                        "category": cat,
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["schema"].get("properties", {}),
                        "required_params": t["schema"].get("required", []),
                    })
            selection_prompt = ReasoningPrompts.TOOL_SELECTION.format(
                current_step=json.dumps(step, ensure_ascii=False)[:1200],
                available_tools=json.dumps(_summarize_tools_for_prompt(tools), ensure_ascii=False)[:2000],
                context=json.dumps(_compact_context_for_prompt(), ensure_ascii=False)[:1200],
            )
            try:
                # Heartbeat before long LLM call
                try:
                    logger.debug("STREAM HB status: starting tool selection LLM call")
                    logger.debug("STREAM OUT status: starting tool selection LLM call")
                    yield json.dumps({"type": "status", "phase": "tool_selection", "message": "Selecting best tool..."}) + "\n"
                except Exception:
                    pass
                # Emit a phase-start thinking step for tool selection
                try:
                    yield json.dumps(await self._emit_step(ThinkingPhase.TOOL_SELECTION, "Evaluating tools and parameters for this step...", 0.5, {"phase_start": True})) + "\n"
                except Exception:
                    pass
                # Tool selection: compact
                selection_text = await self.llm.generate(selection_prompt, system_prompt=workflow_system, max_tokens=500)
                selection = await self._parse_json(selection_text)
                if isinstance(selection, dict) and not selection.get("error"):
                    selection = self._validate_schema(ToolSelectionSchema, selection)
            except Exception as e:
                selection = {"error": str(e)}
            try:
                # Stream full tool selection details as content (JSON string)
                sel_content = json.dumps(selection, ensure_ascii=False)
            except Exception:
                sel_content = "{}"
            logger.debug("STREAM OUT thinking_step: tool_selection")
            yield json.dumps(await self._emit_step(ThinkingPhase.TOOL_SELECTION, sel_content, float(selection.get("confidence", 0.5)), selection)) + "\n"

            # Execute selected tool
            tool_name = selection.get("selected_tool") or ""
            tool_category = selection.get("tool_category") or self._infer_category(tools, tool_name)
            tool_params = selection.get("tool_parameters", {})

            # Prepare sanitized parameters for display (hide sensitive data)
            display_params = {}
            for k, v in tool_params.items():
                if k == 'api_key' or 'secret' in k.lower() or 'key' in k.lower():
                    display_params[k] = '***REDACTED***'
                elif k == 'html_content' and v and len(str(v)) > 100:
                    display_params[k] = f"[HTML content, {len(str(v))} characters]"
                elif isinstance(v, str) and len(v) > 200:
                    display_params[k] = v[:200] + "..."
                else:
                    display_params[k] = v

            # Emit a user-friendly tool call message
            try:
                yield json.dumps(await self._emit_step(
                    ThinkingPhase.EXECUTION,
                    f"🔧 Executing tool: {tool_name}\n" +
                    f"Category: {tool_category}\n" +
                    f"Parameters: {json.dumps(display_params, indent=2)}",
                    confidence=0.9
                )) + "\n"
                
                # Enforce correct URL propagation from context for job tools
                try:
                    ctx_url = (self.context.get("job_url") or "").strip()
                    if ctx_url and tool_category == "job_search":
                        if tool_name in ("precheck_job_offer_url", "fetch_url_html_with_pagination"):
                            # Always use the exact detected/provided URL; avoid LLM examples like company.com
                            tool_params["url"] = ctx_url
                            if tool_name == "fetch_url_html_with_pagination" and "max_pages" not in tool_params:
                                tool_params["max_pages"] = 1
                        elif tool_name == "deep_crawl_job_pages":
                            tool_params["start_url"] = ctx_url
                            tool_params.setdefault("max_pages", 10)
                            tool_params.setdefault("max_depth", 2)
                except Exception as e:
                    logger.warning(f"Error processing URL context: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error emitting tool call message: {e}")
                raise

            # Strict behavior: if tool is unknown or missing, do not apply heuristics/fallbacks
            try:
                cv_tool_names = [t.get("name") for t in tools.get("cv_processing", {}).get("tools", [])]
                job_tool_names = [t.get("name") for t in tools.get("job_search", {}).get("tools", [])]
                all_tool_names = set(cv_tool_names + job_tool_names)
                if (not tool_name) or (tool_name not in all_tool_names):
                    meta = {
                        "llm_selected_tool": tool_name,
                        "available_cv_tools": cv_tool_names,
                        "available_job_tools": job_tool_names,
                    }
                    yield json.dumps(await self._emit_step(
                        ThinkingPhase.TOOL_SELECTION,
                        "Unknown tool selected by LLM; aborting plan (strict, no fallback).",
                        0.1,
                        meta,
                    )) + "\n"
                    aborted_plan = True
                    break
            except Exception:
                aborted_plan = True
                break

            # Emit the raw tool call for debugging
            tool_call_msg = {
                "type": "tool_call",
                "tool": tool_name,
                "category": tool_category,
                "params": display_params  # Use sanitized params for display
            }
            yield json.dumps(tool_call_msg) + "\n"

            # Heartbeat while waiting for tool result
            try:
                logger.debug("STREAM HB status: awaiting tool result for %s", tool_name)
                yield json.dumps({"type": "status", "phase": "execution", "message": f"Executing {tool_name}..."}) + "\n"
            except Exception as e:
                logger.warning(f"Error sending status update: {e}")

            result: Dict[str, Any] = {"success": False, "error": "no_tool"}
            start_time = time.time()
            try:
                # Ensure required params are present for known tools
                if tool_name == "process_cv_pdf":
                    if not tool_params.get("file_path") and self.context.get("file_path"):
                        tool_params["file_path"] = self.context.get("file_path")
                # Guard: skip html_to_markdown when input is empty to avoid wasted calls
                if tool_name == "html_to_markdown" and not tool_params.get("html_content"):
                    result = {"success": False, "error": "Empty html_content; skipping html_to_markdown"}
                elif tool_category == "cv_processing" and self.cv_client:
                    result = await self.cv_client.call_tool(tool_name, tool_params)
                elif tool_category == "job_search" and self.job_client:
                    result = await self.job_client.call_tool(tool_name, tool_params)
                else:
                    raise RuntimeError(f"Unknown or unavailable tool category '{tool_category}' for tool '{tool_name}'")
                executed_tools_count += 1
            except Exception as e:
                result = {"error": str(e)}
                logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
                result = {"success": False, "error": f"Tool execution error: {str(e)}"}

            # Format the result for display (for both success and failure)
            display_result = result
            if isinstance(result, dict):
                display_result = {}
                for k, v in result.items():
                    if k == 'html' and v and len(str(v)) > 100:
                        display_result[k] = f"[HTML content, {len(str(v))} characters]"
                    elif isinstance(v, str) and len(v) > 300:
                        display_result[k] = v[:300] + "..."
                    else:
                        display_result[k] = v

            # Calculate execution time
            execution_time = time.time() - start_time

            # Emit a user-friendly tool result message
            result_status = "✅ Success" if result.get('success', False) else "❌ Failed"
            yield json.dumps(await self._emit_step(
                ThinkingPhase.EXECUTION,
                f"{result_status} - Tool: {tool_name}\n" +
                f"Status: {result_status}\n" +
                f"Duration: {execution_time:.2f}s\n" +
                f"Result: {json.dumps(display_result, indent=2) if isinstance(display_result, dict) else display_result}",
                confidence=0.9 if result.get('success', False) else 0.5
            )) + "\n"

            # Emit the raw tool result for debugging
            tool_result_msg = {
                "type": "tool_result",
                "tool": tool_name,
                "category": tool_category,
                "result": display_result,
                "duration_seconds": execution_time
            }
            yield json.dumps(tool_result_msg) + "\n"

            exec_success = bool(result.get("success"))
            logger.debug("STREAM OUT thinking_step: execution (%s)", "ok" if exec_success else "fail")

            # Prepare execution details for the next phase
            exec_detail = {
                "tool": tool_name,
                "category": tool_category,
                "parameters": tool_params,
                "success": exec_success,
                "raw_result": result,
            }
            yield json.dumps(await self._emit_step(ThinkingPhase.EXECUTION, json.dumps(exec_detail, ensure_ascii=False), 0.9 if exec_success else 0.3, {"tool": tool_name, "result_summary": str(result)[:200]})) + "\n"

            # Simplified pipeline: Skip LLM-based observation. Assume completion if we have usable results.
            observation = {
                "execution_success": bool(exec_success),
                "output_quality": "good" if exec_success else "poor",
                "key_findings": [],
                "issues_detected": [] if exec_success else ["tool_failed"],
                "next_action": "complete" if exec_success else "continue",
                "confidence": 0.7 if exec_success else 0.3,
            }
            try:
                obs_content = json.dumps(observation, ensure_ascii=False)
            except Exception:
                obs_content = "{}"
            logger.debug("STREAM OUT thinking_step: observation (synthetic)")
            yield json.dumps(await self._emit_step(ThinkingPhase.OBSERVATION, obs_content, float(observation.get("confidence", 0.5)), observation)) + "\n"

            all_results.append({
                "step": step,
                "tool_selection": selection,
                "result": result,
                "tool_category": tool_category,
            })

            # Post-process HTML results from job tools -> Markdown evidence (no outer try/except)
            if exec_success and tool_category == "job_search" and self.job_client:
                pages: List[Dict[str, Any]] = []
                rdata = result.get("result") or {}
                if isinstance(rdata, dict):
                    if tool_name in ("fetch_url_html_with_pagination", "deep_crawl_job_pages"):
                        pages = rdata.get("pages", []) if isinstance(rdata.get("pages"), list) else []
                # Prefer job-related pages and limit to 3 conversions to control tokens
                def pick_pages(pages_list):
                    job_pages = [p for p in pages_list if isinstance(p, dict) and p.get("status") == "success" and p.get("html") and p.get("is_job_related")]
                    others = [p for p in pages_list if isinstance(p, dict) and p.get("status") == "success" and p.get("html") and not p.get("is_job_related")]
                    picked = (job_pages + others)[:3]
                    return picked
                picked_pages = pick_pages(pages)
                markdown_entries: List[Dict[str, Any]] = []
                for p in picked_pages:
                    try:
                        md_res = await self.job_client.call_tool("html_to_markdown", {"html_content": p.get("html", "")})
                        if md_res.get("success"):
                            md = (md_res.get("result") or {}).get("markdown", "")
                            entry = {
                                "url": p.get("url"),
                                "depth": p.get("depth"),
                                "is_job_page": p.get("is_job_related") or p.get("is_job_page"),
                                "markdown": md,
                                "markdown_length": len(md),
                            }
                            markdown_entries.append(entry)
                    except Exception:
                        continue
                if markdown_entries:
                    # store in context for final synthesis and UI
                    self.context["markdown_evidence"].extend(markdown_entries)
                    # initialize jobs_found container
                    self.context.setdefault("jobs_found", [])
                    # also attach a compact summary to this result set
                    all_results[-1]["markdown_evidence"] = [
                        {"url": e.get("url"), "depth": e.get("depth"), "is_job_page": e.get("is_job_page"), "markdown_preview": (e.get("markdown") or "")[:400]}
                        for e in markdown_entries
                    ]
                    # Stream markdown page previews
                    for e in markdown_entries:
                        try:
                            md_text = e.get('markdown') or ""
                            logger.debug("STREAM OUT markdown_page: %s", e.get('url'))
                            yield json.dumps({
                                "type": "markdown_page",
                                "url": e.get("url"),
                                "depth": e.get("depth"),
                                "is_job_page": e.get("is_job_page"),
                                "chars": len(md_text),
                                "preview": (md_text[:600] if md_text else "")
                            }) + "\n"
                        except Exception:
                            pass
                    # Batch extract across 2-3 pages per call to reduce LLM requests
                    BATCH_SIZE = 3
                    batch = []
                    for idx_e, e in enumerate(markdown_entries, start=1):
                        batch.append(e)
                        if len(batch) == BATCH_SIZE or idx_e == len(markdown_entries):
                            try:
                                # Build multi-source markdown
                                combined = []
                                for be in batch:
                                    combined.append(f"### SOURCE_BEGIN {be.get('url') or '(unknown)'}\n{be.get('markdown') or ''}\n### SOURCE_END")
                                md_multi = "\n\n".join(combined)
                                jobs = await self._extract_jobs_from_markdown(md_multi)
                                if jobs:
                                    for job in jobs:
                                        # Ground URL if missing using source_url if present, else first batch page url
                                        if not job.get("url"):
                                            fallback = job.get("source_url") or (batch[0].get("url") if batch else None)
                                            if fallback:
                                                job["url"] = fallback
                                        # Ensure source_url exists for grounding
                                        if not job.get("source_url"):
                                            job["source_url"] = (batch[0].get("url") if batch else None) or ""
                                        logger.debug("STREAM OUT job (batched): %s", job.get("title") or job.get("position_name"))
                                        yield json.dumps({
                                            "type": "job",
                                            "data": job
                                        }) + "\n"
                                        try:
                                            self.context["jobs_found"].append(job)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            finally:
                                batch = []
                    logger.debug("STREAM OUT info: converted %d markdown entries", len(markdown_entries))
                    yield json.dumps({"type": "info", "message": f"Converted {len(markdown_entries)} HTML page(s) to markdown for LLM synthesis."}) + "\n"
                    # Finalize immediately after first successful HTML->Markdown conversion batch
                    self.context["evidence_ready"] = True
                    break

            # Update context with notable keys
            if result.get("success") and result.get("result"):
                r = result["result"]
                if isinstance(r, dict):
                    for k in ["corrected_text", "official_website", "possible_job_urls", "pages_crawled"]:
                        if k in r:
                            self.context[f"previous_{k}"] = r[k]

            # Simplified control flow: finalize after first successful tool
            if exec_success:
                self.context["evidence_ready"] = True
                break

        # Enforce: At least one tool must have executed before reflection/final
        if executed_tools_count == 0:
            try:
                yield json.dumps({
                    "type": "error",
                    "phase": "execution",
                    "message": "No tools were executed; stopping before final synthesis."
                }) + "\n"
            except Exception:
                pass
            return

        # Reflection skipped to reduce latency; proceed directly to final synthesis
        try:
            yield json.dumps({"type": "status", "phase": "reflection", "message": "Skipping reflection to respond faster."}) + "\n"
        except Exception:
            pass

        # Final synthesis: always emit a user-facing final message
        try:
            jobs_found = self.context.get("jobs_found") or []
            key_insights = self.context.get("analysis") or {}
            # Construct an evidence digest to keep the most salient, grounded items
            try:
                md_ev = self.context.get("markdown_evidence") or []
                top_md = [
                    {
                        "url": (e.get("url") or "")[:300],
                        "is_job_page": bool(e.get("is_job_page")),
                        "chars": int(e.get("markdown_length") or len(e.get("markdown") or "")),
                        "preview": (e.get("markdown") or "")[:250]
                    }
                    for e in md_ev[:5]
                    if isinstance(e, dict)
                ]
                top_jobs = [
                    {
                        "title": (j.get("title") or j.get("position_name") or "")[:120],
                        "company": (j.get("company") or j.get("company_name") or "")[:120],
                        "location": (j.get("location") or "")[:120],
                        "url": (j.get("url") or j.get("apply_url") or j.get("source_url") or "")[:300]
                    }
                    for j in jobs_found[:6]
                    if isinstance(j, dict)
                ]
                evidence_digest_obj = {"markdown": top_md, "jobs": top_jobs}
            except Exception:
                evidence_digest_obj = {"markdown": [], "jobs": []}
            # Consider multiple signals to determine CV presence
            has_cv = bool(
                self.context.get("previous_corrected_text")
                or self.context.get("corrected_text")
                or (self.context.get("cv_text") and str(self.context.get("cv_text")).strip())
                or (self.context.get("file_path") and str(self.context.get("file_path")).strip())
            )
            matching_intent = bool("match" in (query or "").lower())
            final_prompt = ReasoningPrompts.FINAL_SYNTHESIS.format(
                original_query=(query or "")[:800],
                all_results=json.dumps(all_results, ensure_ascii=False)[:2200],
                key_insights=json.dumps(key_insights, ensure_ascii=False)[:1000],
                jobs_found=json.dumps(jobs_found, ensure_ascii=False)[:1800],
                evidence_digest=json.dumps(evidence_digest_obj, ensure_ascii=False)[:1200],
                has_cv=str(has_cv),
                matching_intent=str(matching_intent),
            )
            # Generate concise final answer for the user (lower temperature for accuracy)
            final_text = await self.llm.generate_with_reasoning(prompt=final_prompt, max_tokens=800, temperature=0.2, force=True)
            yield json.dumps({"type": "final", "final_response": final_text}) + "\n"
        except Exception as e:
            logger.error(f"Error during final synthesis: {str(e)}", exc_info=True)
            # Emit a minimal fallback message so the UI always shows something
            fallback = "I’ve finished processing. If you expected results, please try again or provide a direct job URL or clarify the request."
            yield json.dumps({"type": "final", "final_response": fallback, "error": str(e)}) + "\n"

        # End of reflection block
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a chat completion with advanced rate limiting and automatic retry.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            Dict with response chunks or final result
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        try:
            # Convert messages to prompt if needed
            prompt = messages[-1]['content'] if messages else ""
            system_message = next(
                (msg['content'] for msg in messages if msg.get('role') == 'system'),
                None
            )
            
            # Use generate_with_reasoning to handle rate limiting and simple queries
            response = await self.llm.generate_with_reasoning(
                prompt=prompt,
                system_prompt=system_message,
                max_tokens=max_tokens,
                model=model,
                temperature=temperature,
                force=kwargs.get("force", False),
                **kwargs
            )
            
            if stream:
                async for chunk in self._stream_response(response):
                    yield chunk
            else:
                yield {
                    "choices": [{"message": {"content": response, "role": "assistant"}}],
                    "usage": {
                        "completion_tokens": len(response.split()),
                        "prompt_tokens": sum(len(m['content'].split()) for m in messages if 'content' in m),
                        "total_tokens": 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}", exc_info=True)
            yield {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                }
            }
                                
    async def _stream_response(self, response: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response chunks as they become available.
        
        Args:
            response: The full response text to be streamed
            
        Yields:
            Dict with response chunks in the format expected by the OpenAI API
        """
        for chunk in response.split('\n'):
            if chunk.strip():
                yield {"choices": [{"delta": {"content": chunk + '\n'}}]}

    def _infer_category(self, tools: Dict[str, Any], tool_name: str) -> str:
        for cat, info in tools.items():
            for t in info.get("tools", []):
                if t.get("name") == tool_name:
                    return cat
        return "unknown"
