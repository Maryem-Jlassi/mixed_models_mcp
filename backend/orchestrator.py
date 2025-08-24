import asyncio
import json
import logging
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import httpx
from datetime import datetime
from pathlib import Path

# Suppress MCP progress notification warnings
logging.getLogger('root').setLevel(logging.ERROR)
from datetime import datetime
import time

# Import the MCP clients
from cv_client import MCPClient as CVMCPClient
from job_client import MCPClient as JobMCPClient
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Raised when the Groq API returns HTTP 429 (rate limited)."""
    pass

class UnifiedLLMHost:
    """
    Unified LLM Host that handles CV processing and job search
    through natural conversation and intent analysis.
    Supports job search via company name OR direct job/career page URL.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cv_mcp_server_script: Optional[str] = None,
        job_mcp_server_script: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            raise ValueError("Groq API key is required. Please set it in config.py")

        self.model_name = model_name or config.GROQ_MODEL_NAME
        self.cv_mcp_server_script = cv_mcp_server_script or "cv_ocr_server.py"
        self.job_mcp_server_script = job_mcp_server_script or "server.py"
        self.timeout = timeout or config.GROQ_TIMEOUT
        self.groq_api_url = config.GROQ_API_URL
        
        self.http_client = None
        self.cv_client = None
        self.job_client = None
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self._cv_client_lock = asyncio.Lock()
        self._job_client_lock = asyncio.Lock()

    def _has_job_signals(self, text: str) -> bool:
        """Heuristic: detect if text likely contains job listings to avoid wasted LLM calls."""
        import re
        if not text:
            return False
        patterns = [
            r"\bapply now\b", r"\bwe'?re hiring\b", r"\bjob(s)?\b", r"\bposition(s)?\b",
            r"\bopening(s)?\b", r"\bcareer(s)?\b", r"\bvacanc(y|ies)\b",
            r"\boffre(s)? d'?emploi\b", r"\brecrutement\b", r"\bposte(s)?\b"
        ]
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False

    async def _post_with_backoff(self, payload: Dict[str, Any], max_retries: int = 0):
        """POST to Groq. If 429, return immediately without retrying."""
        response = await self.http_client.post(self.groq_api_url, json=payload)
        if response.status_code == 429:
            logger.warning("Groq 429 rate limit received. Not retrying; caller should handle gracefully.")
        return response

    async def initialize(self):
        try:
            logger.info("🚀 Initializing Unified LLM Host...")
            self.http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
            )
            await self._check_groq_health()
            self.initialized = True
            logger.info("✅ Unified LLM Host initialized successfully (Clients will be loaded on demand)")
        except Exception as e:
            logger.error(f"Failed to initialize Unified LLM Host: {e}")
            await self._cleanup_on_error()
            raise

    async def get_cv_client(self) -> CVMCPClient:
        """Get the CV MCP client, initializing it if necessary."""
        async with self._cv_client_lock:
            if self.cv_client is None:
                logger.info("📄 Initializing CV OCR MCP client on demand...")
                self.cv_client = CVMCPClient(self.cv_mcp_server_script)
                await self.cv_client.start()
                logger.info("✅ CV OCR MCP client initialized")
            return self.cv_client

    async def get_job_client(self) -> JobMCPClient:
        """Get the Job MCP client, initializing it if necessary."""
        async with self._job_client_lock:
            if self.job_client is None:
                logger.info("🔍 Initializing Job Search MCP client on demand...")
                self.job_client = JobMCPClient(self.job_mcp_server_script)
                await self.job_client.start()
                logger.info("✅ Job Search MCP client initialized")
            return self.job_client

    async def _cleanup_on_error(self):
        try:
            if self.cv_client:
                await self.cv_client.stop()
            if self.job_client:
                await self.job_client.stop()
            if self.http_client:
                await self.http_client.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def shutdown(self):
        logger.info("🔄 Shutting down Unified LLM Host...")
        try:
            if self.cv_client:
                await self.cv_client.stop()
                logger.info("✅ CV MCP client stopped")
            if self.job_client:
                await self.job_client.stop()
                logger.info("✅ Job MCP client stopped")
            if self.http_client:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed")
            self.initialized = False
            logger.info("✅ Unified LLM Host shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _check_groq_health(self):
        try:
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0
            }
            response = await self.http_client.post(
                self.groq_api_url,
                json=test_payload,
                timeout=30
            )
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            logger.info(f"✅ Groq API is accessible with model: {self.model_name}")
        except httpx.HTTPStatusError as e:
            error_text = await self._safe_error_text(e.response)
            logger.error(f"Groq API health check failed with HTTP status {e.response.status_code}: {error_text}")
            raise Exception(f"Groq API not accessible: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Groq API health check failed due to a request error: {type(e).__name__} - {e}")
            raise Exception(f"Could not connect to Groq API. Please check your network connection and firewall settings.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Groq API health check: {e}", exc_info=True)
            raise

    async def process_unified_request(self, request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Main coordinator that processes uploads, analyzes intent, and routes requests.
        This is the single entry point for all user requests, streaming all responses.
        """
        session_id = request_data.get("session_id")
        user_message = request_data.get("message", "")
        file_path = request_data.get("file_path")

        if not session_id:
            yield json.dumps({"type": "error", "message": "Session ID is missing."}) + "\n"
            return

        # Initialize session context if it's a new session
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {"cv_text": None}
            logger.info(f"Initialized new session context for {session_id}")

        session_context = self.session_contexts[session_id]

        try:
            # Step 1: Process CV if a new one is uploaded (skip if same CV already cached for this session)
            if file_path:
                last_path = session_context.get('last_file_path')
                if session_context.get('cv_text') and last_path == file_path:
                    # Skip reprocessing and use cached text
                    logger.info(f"Using cached CV for session {session_id}; skipping reprocessing")
                    yield json.dumps({"type": "status", "message": "Using cached CV from this session."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "message": "Processing uploaded CV..."}) + "\n"
                    try:
                        cv_client = await self.get_cv_client()
                        result = await cv_client.process_cv_pdf(file_path=file_path)
                        
                        # Handle nested response structure from CV OCR server
                        if isinstance(result, dict) and 'result' in result and isinstance(result['result'], dict):
                            # New format: {'result': {'text': '...', 'details': {...}}}
                            cv_text = result['result'].get('text', '')
                            details = result['result'].get('details', {})
                        else:
                            # Fallback to direct fields
                            cv_text = result.get('text', '')
                            details = result.get('details', {})
                        
                        # Clean and validate the extracted text
                        cv_text = cv_text.strip()
                        logger.debug(f"Extracted CV text (first 200 chars): {cv_text[:200] if cv_text else 'EMPTY'}")
                        
                        if not cv_text or len(cv_text) < 20:  # Check for meaningful content
                            logger.error(f"CV text extraction failed. Text length: {len(cv_text) if cv_text else 0}")
                            logger.error(f"Full response from CV OCR: {result}")
                            raise ValueError("Failed to extract meaningful text from CV.")
                            
                        # Store the extracted text and details in session context
                        session_context['cv_text'] = cv_text
                        session_context['cv_processing_details'] = details
                        session_context['last_file_path'] = file_path
                        logger.info(f"Successfully processed and cached CV for session {session_id}")
                        yield json.dumps({"type": "status", "message": "CV processed successfully."}) + "\n"
                    except Exception as e:
                        logger.error(f"CV processing failed for session {session_id}: {e}")
                        yield json.dumps({"type": "error", "message": f"CV processing failed: {e}"}) + "\n"
                        return

            # Step 2: Analyze user intent
            yield json.dumps({"type": "status", "message": "Analyzing your request..."}) + "\n"
            has_cv = bool(session_context.get('cv_text'))
            intent_analysis = await self._analyze_user_intent(user_message, has_cv)
            logger.info(f"Intent analysis for session {session_id}: {json.dumps(intent_analysis)}")
            yield json.dumps({"type": "intent", "data": intent_analysis}) + "\n"

            # Step 3: Route to the appropriate workflow
            workflow = intent_analysis.get("workflow", "general")
            has_cv = bool(session_context.get('cv_text'))
            company_names = intent_analysis.get("company_names", [])
            urls = intent_analysis.get("urls", [])
            
            logger.info(f"🎯 Workflow routing - Original: {workflow}, Has CV: {has_cv}, Companies: {company_names}, URLs: {urls}")
            # Heuristic upgrade: if a CV is present and we have a job source (company or URL), force comparison
            if has_cv and (company_names or urls):
                if workflow != "comparison":
                    logger.info("🔧 Upgrading workflow to 'comparison' because CV is present and a job source was detected")
                workflow = "comparison"
            logger.info(f"🚦 Final workflow decision: {workflow}")
            
            if workflow == "cv_analysis":
                async for chunk in self._handle_cv_workflow(request_data, intent_analysis):
                    yield chunk
            elif workflow == "comparison":
                # Ensure session context is passed to comparison workflow
                request_data["session_context"] = session_context
                logger.info(f"🚀 About to start comparison workflow iteration")
                async for chunk in self._handle_comparison_workflow(request_data, intent_analysis):
                    logger.debug(f"📤 Yielding chunk: {chunk[:100] if isinstance(chunk, str) else str(chunk)[:100]}")
                    yield chunk
                logger.info(f"✅ Comparison workflow iteration completed")
            elif workflow == "job_search":
                async for chunk in self._handle_job_workflow(request_data, intent_analysis):
                    yield chunk
            else: # General conversation
                async for chunk in self._generate_general_response(user_message, intent_analysis):
                    yield chunk

        except Exception as e:
            logger.error(f"Error in unified request for session {session_id}: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": f"An unexpected error occurred: {e}"}) + "\n"

    async def _handle_cv_workflow(self, request_data: Dict[str, Any], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        session_id = request_data.get("session_id")
        user_message = request_data.get("message", "")
        cv_text = self.session_contexts.get(session_id, {}).get('cv_text')

        if not cv_text:
            yield json.dumps({"type": "result", "message": "Please upload your CV first to get an analysis."}) + "\n"
            return

        yield json.dumps({"type": "status", "message": "Analyzing your CV..."}) + "\n"
        async for chunk in self._generate_cv_analysis_response(user_message, cv_text, intent):
            yield chunk

    async def _handle_comparison_workflow(self, request_data: Dict[str, Any], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        user_message = request_data.get("message", "")
        session_context = request_data.get("session_context", {})
        cv_text = session_context.get("cv_text", "")
        company_names = intent.get("company_names", [])
        job_urls = intent.get("urls", [])

        logger.info(f"🔄 Starting comparison workflow - CV: {bool(cv_text)}, Companies: {company_names}, URLs: {job_urls}")

        if not cv_text:
            yield json.dumps({"type": "error", "message": "No CV found. Please upload a CV first to compare with job opportunities."}) + "\n"
            return

        if not company_names and not job_urls:
            yield json.dumps({"type": "result", "message": "Please specify a company name or job URL to search for opportunities."}) + "\n"
            return

        all_jobs = []
        job_client = await self.get_job_client()

        # Process direct URLs first (if any) using the unified job URL processor
        if job_urls:
            yield json.dumps({"type": "status", "message": f"Processing {len(job_urls)} job URL(s) with deep crawl..."}) + "\n"
            for url in job_urls:
                try:
                    url_jobs = []
                    async for chunk in self._process_job_url(url):
                        if isinstance(chunk, str):
                            # Forward stream events as-is (includes markdown_page, job, status, error)
                            yield chunk
                        elif isinstance(chunk, list):
                            url_jobs.extend(chunk)
                    if url_jobs:
                        all_jobs.extend(url_jobs)
                        yield json.dumps({"type": "status", "message": f"Found {len(url_jobs)} job(s) from {url}"}) + "\n"
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}", exc_info=True)
                    yield json.dumps({"type": "status", "message": f"Skipping URL due to error: {e}"}) + "\n"
            if not all_jobs:
                yield json.dumps({"type": "status", "message": "No jobs found from provided URL(s)."}) + "\n"
                return
        
        # Process company names only if no URLs provided
        elif company_names:
            for name in company_names:
                yield json.dumps({"type": "status", "message": f"Searching for jobs at {name}..."}) + "\n"
                try:
                    website_info = await job_client.call_tool('get_official_website_and_generate_job_urls', {'company_name': name})
                    # Unwrap and normalize response
                    payload = {}
                    if isinstance(website_info, dict):
                        # Timeout handling
                        if website_info.get('success') is False and website_info.get('error') == 'timeout':
                            yield json.dumps({"type": "status", "message": f"Timed out searching official website for {name}. Skipping."}) + "\n"
                            continue
                        payload = website_info.get('result', website_info)
                    elif isinstance(website_info, str):
                        try:
                            parsed = json.loads(website_info)
                            payload = parsed.get('result', parsed) if isinstance(parsed, dict) else {}
                        except Exception:
                            payload = {}
                    # Try deep crawl from official website FIRST
                    found_from_crawl = []
                    official_site = payload.get('official_website')
                    if isinstance(official_site, str) and official_site.strip():
                        try:
                            yield json.dumps({"type": "status", "message": f"Deep crawling {name}'s site first..."}) + "\n"
                            crawl_res = await job_client.call_tool('deep_crawl_job_pages', {
                                'start_url': official_site,
                                'max_pages': 12,
                                'max_depth': 2,
                                'delay_between_requests': 1.0
                            })
                            # Timeout handling
                            if isinstance(crawl_res, dict) and crawl_res.get('success') is False and crawl_res.get('error') == 'timeout':
                                yield json.dumps({"type": "status", "message": "Deep crawl timed out. Trying direct pages..."}) + "\n"
                                raise Exception("deep_crawl_timeout")
                            crawl_payload = crawl_res.get('result', crawl_res) if isinstance(crawl_res, dict) else {}
                            pages = crawl_payload.get('pages', []) if isinstance(crawl_payload, dict) else []
                            yield json.dumps({"type": "status", "message": f"Crawled {len(pages)} page(s). Extracting job listings..."}) + "\n"

                            for p in pages:
                                html_content = ""
                                if isinstance(p, dict):
                                    for field in ['html', 'content', 'html_content', 'body']:
                                        if p.get(field):
                                            html_content = p.get(field, '')
                                            break
                                if not html_content or len(html_content) < 50:
                                    continue

                                # Convert HTML to markdown
                                md_res = await job_client.call_tool('html_to_markdown', {'html_content': html_content})
                                if isinstance(md_res, dict) and md_res.get('success') is False and md_res.get('error') == 'timeout':
                                    yield json.dumps({"type": "status", "message": "Markdown conversion timed out for a page. Skipping page."}) + "\n"
                                    continue
                                md_payload = md_res.get('result', md_res) if isinstance(md_res, dict) else {}
                                if not md_payload.get('success'):
                                    continue
                                markdown_text = md_payload.get('markdown', '')
                                if not markdown_text:
                                    continue

                                extracted = await self._extract_jobs_from_markdown(markdown_text)
                                if extracted:
                                    total = len(extracted)
                                    for idx, job in enumerate(extracted, start=1):
                                        yield json.dumps({
                                            'type': 'job',
                                            'index': idx,
                                            'total': total,
                                            'data': job
                                        }) + "\n"
                                    found_from_crawl.extend(extracted)

                                if len(found_from_crawl) >= 5:
                                    break

                            if found_from_crawl:
                                all_jobs.extend(found_from_crawl)
                                yield json.dumps({"type": "status", "message": f"Found {len(found_from_crawl)} job(s) via deep crawl."}) + "\n"
                                continue  # Go to next company
                            else:
                                yield json.dumps({"type": "status", "message": "Deep crawl did not yield job listings. Trying direct pages..."}) + "\n"
                        except Exception as e:
                            logger.error(f"Deep crawl (primary) failed for {name}: {e}")
                            yield json.dumps({"type": "status", "message": "Deep crawl error. Trying direct pages..."}) + "\n"

                    potential_urls = []
                    for key in ['job_urls', 'urls', 'career_urls', 'job_pages', 'possible_job_urls']:
                        vals = payload.get(key)
                        if isinstance(vals, list):
                            potential_urls.extend([u for u in vals if isinstance(u, str) and u.strip()])
                        elif isinstance(vals, str) and vals.strip():
                            potential_urls.append(vals.strip())
                    if not potential_urls:
                        base = payload.get('official_website')
                        if isinstance(base, str) and base.strip():
                            base = base.rstrip('/')
                            potential_urls.extend([
                                f"{base}/careers", f"{base}/jobs", f"{base}/career",
                                f"{base}/recrutement", f"{base}/emploi", f"{base}/opportunities",
                                f"{base}/careers/our-job-offers.html"
                            ])
                    # Dedup
                    seen = set()
                    potential_urls = [u for u in potential_urls if not (u in seen or seen.add(u))]
                    if not potential_urls:
                        yield json.dumps({"type": "status", "message": f"Could not find job pages for {name}."}) + "\n"
                        continue
                    
                    yield json.dumps({"type": "status", "message": f"Found {len(potential_urls)} potential job pages for {name}."}) + "\n"
                    
                    # Process each URL for this company
                    found_for_company = False
                    for i, url in enumerate(potential_urls, 1):
                        yield json.dumps({"type": "status", "message": f"Processing job page {i}/{len(potential_urls)}..."}) + "\n"
                        url_jobs = []
                        async for chunk in self._process_job_url(url):
                            if isinstance(chunk, str):
                                yield chunk
                            elif isinstance(chunk, list):
                                url_jobs.extend(chunk)
                        
                        if url_jobs:
                            all_jobs.extend(url_jobs)
                            yield json.dumps({"type": "status", "message": f"Found {len(url_jobs)} job(s) on this page"}) + "\n"
                            found_for_company = True
                            break  # Stop after finding jobs from first successful URL

                    # Deep crawl was already attempted first; no fallback here anymore
                    if not found_for_company:
                        yield json.dumps({"type": "status", "message": "No jobs found on direct pages after deep crawl."}) + "\n"

                except Exception as e:
                    logger.error(f"Error getting jobs for company {name}: {e}")
                    yield json.dumps({"type": "error", "message": f"An error occurred while searching for jobs at {name}."}) + "\n"

        if not all_jobs:
            # If we started with no URLs and no companies, explicitly prompt the user
            if not job_urls and not company_names:
                yield json.dumps({"type": "result", "message": "Please provide a job page URL or company name to compare with your CV."}) + "\n"
                return
            # Graceful fallback: inform the user and still provide an LLM response for guidance
            yield json.dumps({"type": "status", "message": "I couldn't find valid job postings from the provided sources."}) + "\n"
            yield json.dumps({"type": "status", "message": "Providing guidance based on your CV and query..."}) + "\n"
            async for chunk in self._generate_general_response(user_message, intent):
                yield chunk
            return

        yield json.dumps({"type": "status", "message": f"Found {len(all_jobs)} total jobs. Generating comparison..."}) + "\n"
        yield json.dumps({"type": "result_summary", "data": all_jobs}) + "\n"

        # Generate the final comparison response
        yield json.dumps({"type": "status", "message": "Analyzing job matches with your CV..."}) + "\n"
        async for chunk in self._generate_cv_job_matching(user_message, cv_text, all_jobs, intent):
            yield chunk

    async def _handle_job_workflow(self, request_data: Dict[str, Any], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        user_message = request_data.get("message", "")
        company_names = intent.get("company_names", [])
        job_urls = intent.get("urls", [])

        if not company_names and not job_urls:
            yield json.dumps({"type": "result", "message": "What company or job page are you interested in?"}) + "\n"
            return

        all_jobs = []
        
        # If we have direct URLs, process them first and skip company name lookup
        if job_urls:
            yield json.dumps({"type": "status", "message": f"Processing {len(job_urls)} direct URL(s)..."}) + "\n"
            for url in job_urls:
                async for chunk in self._process_job_url(url):
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, list):
                        all_jobs.extend(chunk)
            
            # If we found jobs from direct URLs, we're done
            if all_jobs:
                yield json.dumps({"type": "status", "message": f"Found {len(all_jobs)} jobs from the provided URLs."}) + "\n"
                yield json.dumps({"type": "result_summary", "data": all_jobs}) + "\n"
                async for chunk in self._generate_job_search_response(user_message, all_jobs, intent):
                    yield chunk
                return
        
        # Only process company names if no valid URLs were found
        if company_names:
            job_client = await self.get_job_client()
            for name in company_names:
                yield json.dumps({"type": "status", "message": f"Searching for jobs at {name}..."}) + "\n"
                try:
                    website_info = await job_client.call_tool('get_official_website_and_generate_job_urls', {'company_name': name})
                    # Unwrap nested result if present and normalize keys. Some MCP clients may return a JSON string.
                    payload = {}
                    if isinstance(website_info, dict):
                        if website_info.get('success') is False and website_info.get('error') == 'timeout':
                            yield json.dumps({"type": "status", "message": f"Timed out searching official website for {name}. Skipping."}) + "\n"
                            continue
                        payload = website_info.get('result', website_info)
                    elif isinstance(website_info, str):
                        try:
                            parsed = json.loads(website_info)
                            payload = parsed.get('result', parsed) if isinstance(parsed, dict) else {}
                        except Exception:
                            payload = {}
                    # Log discovered keys for diagnostics
                    try:
                        logger.info(f"Company URL discovery payload keys for {name}: {list(payload.keys())[:10]}")
                    except Exception:
                        pass
                    potential_urls = []
                    for key in ['job_urls', 'urls', 'career_urls', 'job_pages', 'possible_job_urls']:
                        vals = payload.get(key)
                        if isinstance(vals, list):
                            potential_urls.extend([u for u in vals if isinstance(u, str) and u.strip()])
                        elif isinstance(vals, str):
                            # Some servers might return a single URL string
                            if vals.strip():
                                potential_urls.append(vals.strip())
                    # Fallback: if official_website present, synthesize common career paths
                    if not potential_urls:
                        base = payload.get('official_website')
                        if isinstance(base, str) and base.strip():
                            base = base.rstrip('/')
                            synthesized = [
                                f"{base}/careers", f"{base}/jobs", f"{base}/career",
                                f"{base}/recrutement", f"{base}/emploi", f"{base}/opportunities"
                            ]
                            potential_urls.extend(synthesized)
                    # Deduplicate while preserving order
                    seen = set()
                    potential_urls = [u for u in potential_urls if not (u in seen or seen.add(u))]
                    if not potential_urls:
                        yield json.dumps({"type": "status", "message": f"Could not find job pages for {name}."})
                        continue
                    
                    yield json.dumps({"type": "status", "message": f"Found {len(potential_urls)} potential job pages for {name}. Processing..."}) + "\n"
                    for url in potential_urls:
                        async for chunk in self._process_job_url(url):
                            if isinstance(chunk, str):
                                yield chunk
                            elif isinstance(chunk, list):
                                all_jobs.extend(chunk)

                except Exception as e:
                    logger.error(f"Error getting jobs for company {name}: {e}")
                    yield json.dumps({"type": "error", "message": f"An error occurred while searching for jobs at {name}."})

        if not all_jobs:
            yield json.dumps({"type": "result", "message": "I couldn't find any job postings. Please try another company or URL."}) + "\n"
            return

        yield json.dumps({"type": "status", "message": f"Found {len(all_jobs)} total jobs. Preparing your results..."}) + "\n"
        yield json.dumps({"type": "result_summary", "data": all_jobs}) + "\n"

        async for chunk in self._generate_job_search_response(user_message, all_jobs, intent):
            yield chunk

    async def _process_job_url(self, url: str) -> AsyncGenerator[Union[str, List[Dict]], None]:
        """
        Processes a single URL through the full job extraction pipeline and streams status.
        Yields status strings and a final list of extracted job dictionaries.
        """
        job_client = await self.get_job_client()
        try:
            # 0. Deep crawl first (make server.py the base strategy)
            yield json.dumps({"type": "status", "message": f"Deep crawling site starting from: {url[:70]} ..."}) + "\n"
            try:
                crawl_res = await job_client.call_tool('deep_crawl_job_pages', {
                    'start_url': url,
                    'max_pages': 12,
                    'max_depth': 2,
                    'delay_between_requests': 1.0
                })
                if isinstance(crawl_res, dict) and crawl_res.get('success') is False and crawl_res.get('error') == 'timeout':
                    yield json.dumps({"type": "status", "message": "Deep crawl timed out. Proceeding with direct fetch..."}) + "\n"
                    raise Exception("deep_crawl_timeout")
                crawl_payload = crawl_res.get('result', crawl_res) if isinstance(crawl_res, dict) else {}
                pages = crawl_payload.get('pages', []) if isinstance(crawl_payload, dict) else []
                yield json.dumps({"type": "status", "message": f"Crawled {len(pages)} page(s). Extracting job listings..."}) + "\n"

                # Surface crawler statistics and per-page issues (robots.txt, errors) to the frontend
                try:
                    stats = crawl_payload.get('statistics', {}) if isinstance(crawl_payload, dict) else {}
                    skipped_count = stats.get('pages_skipped', 0)
                    failed_count = stats.get('pages_failed', 0)
                    if skipped_count:
                        yield json.dumps({"type": "status", "message": f"Note: {skipped_count} page(s) were skipped (e.g., robots.txt or skip patterns)."}) + "\n"
                    if failed_count:
                        yield json.dumps({"type": "status", "message": f"Note: {failed_count} page(s) failed to crawl."}) + "\n"
                except Exception:
                    pass

                # Emit detailed reasons for skipped/failed pages
                for p in pages:
                    if not isinstance(p, dict):
                        continue
                    status = p.get('status')
                    if status == 'skipped':
                        reason = p.get('reason') or 'Skipped'
                        url_msg = p.get('url', '')[:120]
                        yield json.dumps({"type": "status", "message": f"Skipped: {url_msg} — {reason}"}) + "\n"
                    elif status == 'failed':
                        err = p.get('error') or 'Unknown error'
                        url_msg = p.get('url', '')[:120]
                        yield json.dumps({"type": "status", "message": f"Failed: {url_msg} — {err}"}) + "\n"

                # Aggregate markdown from base and subpages so LLM can link details across pages
                md_pages: List[Dict[str, Any]] = []
                for p in pages:
                    if not isinstance(p, dict):
                        continue
                    html_content = ""
                    for field in ['html', 'content', 'html_content', 'body']:
                        if p.get(field):
                            html_content = p.get(field, '')
                            break
                    if not html_content or len(html_content) < 50:
                        continue
                    md_res = await job_client.call_tool('html_to_markdown', {'html_content': html_content})
                    if isinstance(md_res, dict) and md_res.get('success') is False and md_res.get('error') == 'timeout':
                        yield json.dumps({"type": "status", "message": "Markdown conversion timed out for a page. Skipping page."}) + "\n"
                        continue
                    md_payload = md_res.get('result', md_res) if isinstance(md_res, dict) else {}
                    if not md_payload.get('success'):
                        continue
                    markdown_text = (md_payload.get('markdown') or '').strip()
                    if not markdown_text:
                        continue
                    md_pages.append({
                        'url': p.get('url', ''),
                        'depth': p.get('depth', None),
                        'is_job_page': p.get('is_job_page', None),
                        'markdown': markdown_text
                    })
                    # Stream a compact preview of each markdown page for the UI
                    try:
                        preview = markdown_text[:800]
                        yield json.dumps({
                            'type': 'markdown_page',
                            'url': p.get('url', ''),
                            'depth': p.get('depth', None),
                            'is_job_page': p.get('is_job_page', None),
                            'chars': len(markdown_text),
                            'preview': preview
                        }) + "\n"
                    except Exception:
                        pass

                # Partition base vs subpages, limit sizes to control token usage
                base_md = ''
                base_url = url
                for item in md_pages:
                    if item.get('depth') == 0:
                        base_md = item['markdown']
                        base_url = item.get('url') or base_url
                        break
                if not base_md and md_pages:
                    base_md = md_pages[0]['markdown']
                    base_url = md_pages[0].get('url') or base_url

                # Keep a curated set of subpages with likely job details first
                def _is_probable_detail(u: str) -> bool:
                    if not isinstance(u, str):
                        return False
                    u = u.lower()
                    pats = [
                        r"/(offres-emploi|jobs?|careers?|emploi|recrutement)/\d{3,}(/|$)",
                        r"/(job|offer|offre)[-_/]?\d{2,}(/|-)",
                        r"/(apply|postuler)(/|$)",
                    ]
                    import re as _re
                    return any(_re.search(p, u) for p in pats)

                subpages_sorted = sorted(
                    [i for i in md_pages if i.get('url') != base_url],
                    key=lambda x: (0 if _is_probable_detail(x.get('url','')) else 1, x.get('depth') if x.get('depth') is not None else 99)
                )

                # Truncate markdown to reasonable sizes
                def _truncate(md: str, limit: int = 6000) -> str:
                    return md if len(md) <= limit else md[:limit]

                base_md_trunc = _truncate(base_md, 8000)
                sub_md_infos = []
                for itm in subpages_sorted[:12]:
                    sub_md_infos.append({
                        'url': itm.get('url', ''),
                        'markdown': _truncate(itm.get('markdown', ''), 4000)
                    })

                # Ask LLM to extract jobs with cross-page linking
                yield json.dumps({"type": "status", "message": "Linking details across base and subpages..."}) + "\n"
                try:
                    jobs_multi = await self._extract_jobs_from_multi_markdown(base_md_trunc, sub_md_infos, base_url=base_url)
                except RateLimitError as e:
                    msg = (
                        "Too many requests while extracting jobs (rate limited). "
                        "Please provide the exact job posting URL to avoid broad crawling.\n"
                        f"Details: {str(e)[:500]}"
                    )
                    yield json.dumps({"type": "error", "message": msg}) + "\n"
                    return
                if jobs_multi:
                    total = len(jobs_multi)
                    for idx, job in enumerate(jobs_multi, start=1):
                        yield json.dumps({'type': 'job','index': idx,'total': total,'data': job}) + "\n"
                    yield json.dumps({"type": "status", "message": f"Found {len(jobs_multi)} job(s) via multi-page extraction."}) + "\n"
                    yield jobs_multi
                    return

                # Fallback: per-page extraction (existing behavior)
                found_from_crawl: List[Dict[str, Any]] = []
                for item in md_pages:
                    try:
                        extracted = await self._extract_jobs_from_markdown(item['markdown'])
                    except RateLimitError as e:
                        msg = (
                            "Too many requests while extracting from a subpage (rate limited). "
                            "Consider providing the exact job posting URL so I don't need to crawl multiple pages.\n"
                            f"Details: {str(e)[:500]}"
                        )
                        yield json.dumps({"type": "error", "message": msg}) + "\n"
                        return
                    if extracted:
                        total = len(extracted)
                        for idx, job in enumerate(extracted, start=1):
                            yield json.dumps({'type': 'job','index': idx,'total': total,'data': job}) + "\n"
                        found_from_crawl.extend(extracted)
                    if len(found_from_crawl) >= 5:
                        break
                if found_from_crawl:
                    yield json.dumps({"type": "status", "message": f"Found {len(found_from_crawl)} job(s) via deep crawl."}) + "\n"
                    yield found_from_crawl
                    return
                else:
                    yield json.dumps({"type": "status", "message": "Deep crawl found no jobs. Falling back to direct page processing..."}) + "\n"
            except Exception as e:
                logger.error(f"Deep crawl primary step failed for {url}: {e}")
                yield json.dumps({"type": "status", "message": "Deep crawl encountered an error. Trying direct page processing..."}) + "\n"

            # 1. Pre-check URL
            yield json.dumps({"type": "status", "message": f"Checking URL: {url[:70]}..."}) + "\n"
            precheck_result = await job_client.call_tool('precheck_job_offer_url', {'url': url})
            if isinstance(precheck_result, dict) and precheck_result.get('success') is False and precheck_result.get('error') == 'timeout':
                yield json.dumps({"type": "status", "message": "Precheck timed out. Aborting this URL."}) + "\n"
                return
            # Unwrap nested result structure if present
            precheck_payload = precheck_result.get('result', precheck_result) if isinstance(precheck_result, dict) else {}
            if not precheck_payload.get('is_offer_page', False):
                reason = precheck_payload.get('reason', 'Not a valid job page.')
                # If robots.txt blocks, surface a detailed error and stop.
                try:
                    import urllib.parse as _urlparse
                    parsed = _urlparse.urlparse(url or '')
                    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt" if parsed.scheme and parsed.netloc else "robots.txt"
                except Exception:
                    robots_url = "robots.txt"
                if 'robots' in (reason or '').lower():
                    detailed = (
                        f"Error: Access to this page is blocked by the site's robots.txt policy, so we did not crawl it.\n"
                        f"Site: {parsed.netloc if 'parsed' in locals() else ''}\n"
                        f"Robots policy: {robots_url}\n\n"
                        f"What you can try:\n"
                        f"- Provide a direct job posting URL that is publicly accessible.\n"
                        f"- Try another section of the site (e.g., a dedicated jobs board subdomain) that allows crawling."
                    )
                    yield json.dumps({"type": "error", "message": detailed}) + "\n"
                    return

                # Soft fallback: proceed if URL strongly suggests jobs
                url_lower = (url or '').lower()
                strong_signals = any(sig in url_lower for sig in [
                    '/jobs', '/careers', '/career', '/recrutement', '/emploi', '/opportunities'
                ])
                if strong_signals:
                    yield json.dumps({"type": "status", "message": f"Precheck: {reason}. Proceeding due to strong URL signals."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "message": f"Skipping URL: {reason}"}) + "\n"
                    return

            # 2. Fetch HTML
            yield json.dumps({"type": "status", "message": "Fetching page content..."}) + "\n"
            html_result = await job_client.call_tool('fetch_url_html_with_pagination', {'url': url})
            if isinstance(html_result, dict) and html_result.get('success') is False and html_result.get('error') == 'timeout':
                yield json.dumps({"type": "status", "message": "HTML fetch timed out. Aborting this URL."}) + "\n"
                return
            html_payload = html_result.get('result', html_result) if isinstance(html_result, dict) else {}
            if not html_payload.get('success', False):
                yield json.dumps({"type": "status", "message": "Failed to fetch page content."}) + "\n"
                return

            # Extract HTML from pages array
            html_content = ""
            pages = html_payload.get('pages', [])
            for page in pages:
                if isinstance(page, dict):
                    for field in ['html', 'content', 'html_content', 'body']:
                        if page.get(field):
                            html_content = page.get(field, '')
                            break
                    if html_content:
                        break
                elif isinstance(page, str) and len(page) > 100:
                    html_content = page
                    break
            if not html_content:
                yield json.dumps({"type": "status", "message": "Failed to extract HTML from fetched pages."}) + "\n"
                return

            # 3. Convert to Markdown
            yield json.dumps({"type": "status", "message": "Converting page to markdown..."}) + "\n"
            markdown_result = await job_client.call_tool('html_to_markdown', {'html_content': html_content})
            if isinstance(markdown_result, dict) and markdown_result.get('success') is False and markdown_result.get('error') == 'timeout':
                yield json.dumps({"type": "status", "message": "Markdown conversion timed out. Aborting this URL."}) + "\n"
                return
            markdown_payload = markdown_result.get('result', markdown_result) if isinstance(markdown_result, dict) else {}
            if not markdown_payload.get('success', False):
                yield json.dumps({"type": "status", "message": "Failed to convert page to markdown."}) + "\n"
                return
            markdown_text = markdown_payload.get('markdown', '')
            if not markdown_text:
                yield json.dumps({"type": "status", "message": "Empty markdown content after conversion."}) + "\n"
                return

            # 4. Extract jobs from Markdown
            yield json.dumps({"type": "status", "message": "Extracting job details from page... This may take some time."}) + "\n"
            try:
                extracted_jobs = await self._extract_jobs_from_markdown(markdown_text)
            except RateLimitError as e:
                msg = (
                    "Too many requests to the LLM during extraction (429). "
                    "To proceed, please provide the exact job offer URL if you have it, since the system is now trying to infer it from a company name.\n"
                    f"Details: {str(e)[:500]}"
                )
                yield json.dumps({"type": "error", "message": msg}) + "\n"
                return
            if extracted_jobs:
                # Stream each job as it is found
                total = len(extracted_jobs)
                for i, job in enumerate(extracted_jobs, start=1):
                    yield json.dumps({
                        "type": "job",
                        "index": i,
                        "total": total,
                        "data": job
                    }) + "\n"
                yield json.dumps({"type": "status", "message": f"Found {len(extracted_jobs)} jobs from URL."}) + "\n"
                yield extracted_jobs # Yield the final list of jobs
            else:
                yield json.dumps({"type": "status", "message": "No jobs found on this page."}) + "\n"

        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            yield json.dumps({"type": "error", "message": f"Error processing {url}: {e}"}) + "\n"

    async def _extract_jobs_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Uses an LLM to extract structured job information from markdown text."""
        # Duplicate implementation removed; canonical enhanced implementation is defined later in the file.
        # This stub should never be reached because the enhanced version shadows this earlier definition.
        return []
    async def _generate_cv_analysis_response(self, user_message: str, cv_text: str, intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate CV analysis response based on user query and CV content"""
        logger.info("🤖 Generating CV analysis response")
        
        # Check if this might be a mixed query (CV + job search)
        company_names = intent.get("company_names", [])
        urls = intent.get("urls", [])
        
        if company_names or urls:
            # This is actually a comparison query, not pure CV analysis
            yield json.dumps({"type": "status", "message": "Detected job search in your query. Switching to comparison mode..."})
            system_prompt = f"""You are a professional CV analyst. The user has uploaded their CV and is asking about jobs or companies.
            
User query: '{user_message}'
CV content: {cv_text[:8000]}

The user seems interested in companies: {company_names} or URLs: {urls}

Provide a helpful response that:
1. Acknowledges their CV
2. Notes their interest in specific companies/jobs
3. Suggests they can search for jobs at those companies for comparison
4. Offers to analyze their CV in the context of those opportunities"""
        else:
            # Pure CV analysis
            system_prompt = f"""You are an expert CV analyst. The user wants to discuss their CV.
            Their intent is: {intent.get('llm_reasoning', 'Not specified')}.
            
User query: '{user_message}'
CV content: {cv_text[:10000]}

Analyze the CV and respond to their query with:
1. Direct answers to their specific question
2. Professional but conversational tone

Focus on what they asked while providing valuable CV insights."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True
        }

        # Stream the response with proper formatting
        buffer = ""
        async for chunk in self._stream_groq_response(payload):
            try:
                data = json.loads(chunk)
                if data.get('type') == 'response' and 'content' in data:
                    buffer += data['content']
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            yield json.dumps({"type": "response", "content": line + '\n'}) + "\n"
                        buffer = lines[-1]
                    elif len(buffer) > 300:
                        yield json.dumps({"type": "response", "content": buffer}) + "\n"
                        buffer = ""
                else:
                    yield chunk
            except json.JSONDecodeError:
                yield chunk
        
        if buffer.strip():
            yield json.dumps({"type": "response", "content": buffer}) + "\n"

    async def _generate_job_search_response(self, user_message: str, job_results: List[Dict], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate a conversational job search response listing found jobs"""
        
        # Determine if this is a company search or URL search
        company_names = intent.get("company_names", [])
        urls = intent.get("urls", [])
        
        search_context = ""
        if urls:
            search_context = f"from the job page(s) you provided"
        elif company_names:
            company_list = ", ".join(company_names)
            search_context = f"at {company_list}"
        else:
            search_context = "based on your search"
            
        system_prompt = f"""You are a helpful career assistant. The user searched for jobs {search_context} and you found {len(job_results)} job listings.

**Job Results:**
{json.dumps(job_results, indent=2)[:8000]}

**User Query:** {user_message}

Create a conversational response that:
1. Acknowledges their search request naturally
2. Lists the jobs in a clear, engaging format with job titles, companies, and key details
3. Uses bullet points or numbered lists for easy reading
4. Mentions if these are from a specific company or URL they provided
5. Offers to help with CV analysis or comparison if they're interested

Keep the tone professional but conversational, like talking to a career advisor."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True
        }

        # Stream the response with proper formatting
        buffer = ""
        async for chunk in self._stream_groq_response(payload):
            try:
                data = json.loads(chunk)
                if data.get('type') == 'response' and 'content' in data:
                    buffer += data['content']
                    # Send complete lines for better formatting
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            yield json.dumps({"type": "response", "content": line + '\n'})
                        buffer = lines[-1]
                    elif len(buffer) > 300:
                        yield json.dumps({"type": "response", "content": buffer})
                        buffer = ""
                else:
                    yield chunk
            except json.JSONDecodeError:
                yield chunk
        
        if buffer.strip():
            yield json.dumps({"type": "response", "content": buffer})

    async def _generate_cv_job_matching(self, user_message: str, cv_text: str, job_info: List[Dict], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        # Duplicate removed in favor of the later canonical implementation below.
        # This placeholder should not be invoked; the later definition is the source of truth.
        if False:
            yield ""  # Unreachable


    async def _generate_general_response(self, user_message: str, intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        # Duplicate removed in favor of the later canonical implementation (if present) or this one only.
        system_prompt = f"""You are a helpful career and CV assistant. You have analyzed the user's intent and determined this is a general conversation.

**INTENT ANALYSIS CONTEXT:**
- LLM Reasoning: {intent.get('reasoning', 'Not provided')}
- Confidence: {intent.get('confidence', 0.0)}
- Primary Intent: {intent.get('primary_intent', 'general_conversation')}

**Your Instructions:**
1. Acknowledge the user's message in a friendly, conversational tone.
2. If appropriate, gently offer help with CV analysis or job search.
3. Keep your response concise and helpful.

Start your response naturally without explicitly mentioning the intent analysis."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": True
        }

        # Stream using the common helper to keep output format consistent
        async for chunk in self._stream_groq_response(payload):
            yield chunk

    async def _generate_cv_job_matching(self, user_message: str, cv_text: str, job_info: List[Dict], intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Generates a response comparing the CV with job listings.
        Streams the analysis to the frontend as it's generated.
        """
        # Prepare the prompt for the LLM (prioritize Job Posting Description + CV context)
        prompt = f"""You are a precise CV-to-job matching assistant.

PRIMARY FOCUS:
- For each job, base your analysis PRIMARILY on the Job Posting Description field (job['description'] or 'Job Posting Description'), and SECONDARILY on 'requirements_snippet'.
- Consider other metadata (title, sector, contract_type, location, salary, experience, mobility, languages) as supporting signals only.

HOW TO READ THE CV:
- Use the candidate's contextual experience: roles, responsibilities, achievements, projects, domains, and technologies. Avoid superficial keyword matching.
- Extract concrete evidence from the CV (e.g., project names, metrics, tech stacks) to justify matches.

OUTPUT STYLE:
- For each job, provide:
  1) A concise match summary grounded in short quotes/snippets from BOTH the job description and the CV.
  2) A match score 0–100 with 1-sentence rationale.
  3) Key strengths (2–4 bullets) and gaps (2–4 bullets) focused on the job description.
  4) Actionable tailoring tips for the CV (2–4 bullets), prioritizing what the job description emphasizes.
- Be concise and structured. Prefer bullet points.

CONTEXT
CV CONTENT (truncated):
{cv_text[:10000]}

JOB LISTINGS (truncated):
{json.dumps(job_info, indent=2)[:10000]}

USER MESSAGE: {user_message}

Important rules:
- Quote only small, relevant snippets from the job description and the CV to ground claims.
- If a field is 'not mentioned', do not speculate.
- If multiple jobs are similar, still analyze each job separately.
- If languages/experience/mobility constraints conflict with the CV, call them out explicitly.
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a rigorous, concise CV-to-job matching advisor. You prioritize the job's description text and the CV's contextual experience, and you ground your reasoning with short quotes."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True
        }

        # Stream the response directly to the frontend, while buffering for a final full text emission
        collected_text = []
        async for chunk in self._stream_groq_response(payload):
            try:
                # Attempt to capture streamed 'response' content while forwarding
                if isinstance(chunk, str):
                    parsed = json.loads(chunk.strip()) if chunk.strip().startswith('{') else None
                    if isinstance(parsed, dict) and parsed.get('type') == 'response':
                        content = parsed.get('content') or ''
                        collected_text.append(content)
            except Exception:
                # Ignore parsing issues and just forward the chunk
                pass
            # Forward chunk as-is for live updates
            yield chunk

        # Emit a final consolidated result message with the full matching analysis text
        try:
            full_text = ''.join(collected_text).strip()
            if full_text:
                yield json.dumps({"type": "result", "message": full_text}) + "\n"
        except Exception:
            # If consolidation fails, silently skip to avoid breaking the stream
            pass

    async def _extract_jobs_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Extract job listings from markdown content using improved LLM approach."""
        if not markdown_content.strip():
            return []
        
        logger.info(f"🎯 Starting job extraction from {len(markdown_content)} characters of markdown")

        # Early exit: avoid LLM calls if content is tiny and lacks job signals
        if len(markdown_content) < 200 and not self._has_job_signals(markdown_content):
            logger.info("Skipping LLM extraction: content too short and no job signals detected")
            return []
        
        # For large content, use chunked extraction
        if len(markdown_content) > 5000:
            logger.info("Content too large, using chunked extraction")
            return await self._extract_jobs_from_chunks(markdown_content)
        
        # Enhanced extraction prompt with better structure
        prompt = f"""Extract ALL job listings from this markdown content. Return ONLY a JSON array.

JOB IDENTIFICATION IN MARKDOWN:
- Look for job titles in headings (# heading) or emphasized text (**bold** or *italic*)
- Job listings often follow patterns with sections for description, requirements, etc.
- Jobs might be separated by horizontal rules (---) or headers
- Look for key phrases like "Job Title", "Position", "Role", "Opening", "We're hiring", "Apply now"

OUTPUT FORMAT:
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

IMPORTANT RULES:
- Use "not mentioned" for missing fields, NOT null or N/A
- Extract ALL distinct job positions - do NOT limit the number
- If the same job appears multiple times, include it only once
- For each separate job position, create a separate JSON object
- BE GENEROUS in what you consider a job - if it's mentioned at all, include it

MARKDOWN CONTENT:
{markdown_content[:4500]}

JSON OUTPUT:"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a job extraction AI. Extract job listings and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False
        }

        try:
            response = await self._post_with_backoff(payload)
            if response.status_code == 429:
                # Surface the 429 to the caller with any response text
                error_text = await self._safe_error_text(response)
                raise RateLimitError(f"Groq rate limit (429). {error_text}")
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                
                logger.info(f"📝 Raw LLM extraction output: {len(content)} characters")
                
                # Try multiple extraction strategies
                jobs = self._extract_jobs_json(content)
                
                if not jobs:
                    logger.info("No jobs found with standard extraction, trying aggressive method")
                    jobs = await self._aggressive_job_extraction(markdown_content)
                
                # Clean and enhance job data
                cleaned_jobs = []
                for job in jobs:
                    clean_job = self._clean_job_data(job)
                    cleaned_jobs.append(clean_job)
                
                logger.info(f"🎯 Successfully extracted {len(cleaned_jobs)} jobs")
                return cleaned_jobs
            else:
                logger.error(f"Failed to extract jobs: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error extracting jobs from markdown: {e}")
            return []

    async def _extract_jobs_from_chunks(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Extract jobs from large markdown content by processing it in chunks."""
        import re
        
        # Split content into manageable chunks
        chunks = self._chunk_markdown(markdown_content, 3000)
        all_jobs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            extraction_prompt = f"""Extract job listings from this content chunk ({i+1}/{len(chunks)}).
Return ONLY a JSON array of jobs.

Format:
[
  {{
    "title": "Job Title",
    "company": "Company Name",
    "location": "Location if mentioned or 'not mentioned'",
    "description": "Brief description if available or 'not mentioned'"
  }}
]

CONTENT CHUNK:
{chunk[:2500]}

JSON OUTPUT:"""
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a job extraction AI. Extract job listings and return only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": False
            }
            
            # Skip tiny/no-signal chunks
            if len(chunk) < 200 and not self._has_job_signals(chunk):
                logger.info(f"Skipping chunk {i+1}: too short and no job signals")
                continue
            try:
                response = await self._post_with_backoff(payload)
                if response.status_code == 429:
                    error_text = await self._safe_error_text(response)
                    raise RateLimitError(f"Groq rate limit (429). {error_text}")
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                    
                    chunk_jobs = self._extract_jobs_json(content)
                    if chunk_jobs:
                        logger.info(f"Found {len(chunk_jobs)} jobs in chunk {i+1}")
                        # Check for duplicates before adding
                        for job in chunk_jobs:
                            if job.get('title') and not any(j.get('title') == job.get('title') for j in all_jobs):
                                all_jobs.append(job)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
        
        logger.info(f"Total jobs extracted from chunks: {len(all_jobs)}")
        return all_jobs

    async def _extract_jobs_from_multi_markdown(self, base_markdown: str, sub_md_infos: List[Dict[str, Any]], base_url: str = "") -> List[Dict[str, Any]]:
        """Extract jobs using combined context from a base page and multiple subpages.

        The LLM is instructed to consolidate details appearing across pages into unified job objects.
        """
        try:
            if not base_markdown and not sub_md_infos:
                return []

            # Build a concise, structured prompt with page sections
            def trunc(s: str, n: int) -> str:
                return s if len(s) <= n else s[:n]

            base_section = f"""BASE PAGE ({base_url or 'unknown'}):\n{trunc(base_markdown or '', 8000)}\n"""

            sub_sections = []
            for i, info in enumerate(sub_md_infos[:12], start=1):
                u = info.get('url', '')
                md = trunc(info.get('markdown', '') or '', 4000)
                if md:
                    sub_sections.append(f"[SUBPAGE {i}] URL: {u}\n{md}")

            context_blob = base_section + "\n\nSUBPAGES (link info across these):\n" + ("\n\n".join(sub_sections) if sub_sections else "<none>")

            prompt = f"""You are a job extraction AI. Extract ALL distinct job postings by using information across a base page and its subpages.

Rules:
- Consolidate details from different pages that refer to the same job into ONE JSON object.
- Prefer the most specific details from subpages (e.g., description, requirements, salary, contract type).
- If a field is missing, use "not mentioned" (never null/empty).
- Be generous in identifying jobs; if unsure, include it.
- Remove duplicates: same title and company should appear once.

Output strictly as a JSON array of objects with these fields only:
[
  {{
    "title": "Exact job title",
    "company": "Company name or 'not mentioned'",
    "location": "Location or 'not mentioned'",
    "description": "Brief description or 'not mentioned'",
    "salary": "Salary or 'not mentioned'",
    "requirements": "Key qualifications or 'not mentioned'",
    "contrat-type": "Full-time/Part-time/etc. or 'not mentioned'",
    "required_skill": "Key skills or 'not mentioned'",
    "post_date": "Posting date or 'not mentioned'"
  }}
]

CONTENT TO ANALYZE (base + subpages in sections):\n\n{context_blob}\n\nJSON OUTPUT:"""

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You extract job listings and return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 3072,
                "stream": False
            }

            response = await self._post_with_backoff(payload)
            if response.status_code == 429:
                error_text = await self._safe_error_text(response)
                raise RateLimitError(f"Groq rate limit (429). {error_text}")
            if response.status_code != 200:
                logger.error(f"Multi-markdown extraction error: {response.status_code}")
                return []

            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
            jobs = self._extract_jobs_json(content)

            # Clean and deduplicate
            cleaned: List[Dict[str, Any]] = []
            seen = set()
            for job in jobs or []:
                cj = self._clean_job_data(job)
                key = (cj.get('title', '').lower(), cj.get('company', '').lower())
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(cj)
            return cleaned
        except Exception as e:
            logger.error(f"Error in multi-markdown extraction: {e}", exc_info=True)
            return []

    def _chunk_markdown(self, markdown_content: str, max_chars: int = 4000) -> List[str]:
        """Split markdown content into smaller chunks for processing."""
        import re
        
        if len(markdown_content) <= max_chars:
            return [markdown_content]
        
        # Split by sections (headers)
        section_pattern = r'(?=(?:^|\n)#+\s+)'
        sections = re.split(section_pattern, markdown_content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) <= max_chars:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                if len(section) > max_chars:
                    # Break section by paragraphs
                    paragraphs = section.split("\n\n")
                    for paragraph in paragraphs:
                        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                            current_chunk += paragraph + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = paragraph + "\n\n" if len(paragraph) <= max_chars else paragraph[:max_chars]
                else:
                    current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _extract_jobs_json(self, llm_output: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM output with improved parsing."""
        import re
        import json
        
        if not llm_output or not isinstance(llm_output, str):
            return []
            
        llm_output = llm_output.strip()
        
        # Try backtick-wrapped JSON first
        backtick_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', llm_output, re.DOTALL)
        if backtick_match:
            try:
                json_str = backtick_match.group(1).strip()
                fixed_json = self._fix_json_aggressively(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except Exception as e:
                logger.error(f"Backtick JSON parsing failed: {e}")
        
        # Try direct array parsing
        try:
            array_match = re.search(r'\[\s*\{.*?\}(?:,\s*\{.*?\})*\s*\]', llm_output, re.DOTALL)
            if array_match:
                json_str = array_match.group(0).strip()
                fixed_json = self._fix_json_aggressively(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
        except Exception as e:
            logger.error(f"Array parsing failed: {e}")
        
        return []

    def _fix_json_aggressively(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        import re
        
        if not json_str or not isinstance(json_str, str):
            return "[]"
        
        try:
            # Remove markdown code blocks
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            
            # Remove comments
            json_str = re.sub(r'//.*', '', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Fix quotes
            json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Single quote keys
            json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Single quote values
            
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix missing quotes around unquoted keys
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
            
            return json_str.strip()
            
        except Exception as e:
            logger.error(f"JSON fixing failed: {e}")
            return json_str

    async def _aggressive_job_extraction(self, markdown_content: str) -> List[Dict[str, Any]]:
        """Aggressively extract anything that looks like a job from the content."""
        import re
        
        logger.info("Using aggressive job extraction method")

        if len(markdown_content) < 200 and not self._has_job_signals(markdown_content):
            logger.info("Skipping aggressive extraction: content too short and no job signals")
            return []
        
        # Simplified prompt that focuses on just finding job titles
        aggressive_prompt = f"""Look through this content and extract ANY possible job titles mentioned.
Be extremely generous - include anything that might be a job position.
Return ONLY a JSON array with title objects.

Example output:
[
  {{"title": "Software Engineer"}},
  {{"title": "Marketing Manager"}},
  {{"title": "Data Scientist"}}
]

Content:
{markdown_content[:4000]}

OUTPUT (JSON ARRAY ONLY):"""
    
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a job extraction AI. Extract job titles and return only valid JSON."},
                {"role": "user", "content": aggressive_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            response = await self.http_client.post(self.groq_api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                
                job_objects = self._extract_jobs_json(content)
                
                # If still no jobs found, try regex extraction
                if not job_objects:
                    titles = re.findall(r'(?:position|job|title|role):\s*([^\n\.]{3,50})', markdown_content, re.IGNORECASE)
                    job_objects = [{"title": title.strip()} for title in titles if len(title.strip()) > 3]
                
                # Add default fields to all jobs
                enhanced_jobs = []
                for job in job_objects:
                    job_info = {
                        "title": job.get("title", "Unknown Position"),
                        "company": "Unknown Company",
                        "location": "not mentioned",
                        "description": "not mentioned",
                        "salary": "not mentioned",
                        "requirements": "not mentioned",
                        "contrat-type": "not mentioned",
                        "required_skill": "not mentioned",
                        "post_date": "not mentioned"
                    }
                    enhanced_jobs.append(job_info)
                    
                return enhanced_jobs
            else:
                logger.error(f"Aggressive extraction LLM API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Aggressive extraction failed: {e}")
            return []

    def _clean_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize job data."""
        clean_job = {}
        
        # Copy all fields directly
        for key, value in job_data.items():
            if value is None:
                clean_job[key] = "not mentioned"
            elif isinstance(value, list):
                clean_job[key] = ", ".join(str(item) for item in value if item)
            else:
                clean_job[key] = str(value).strip()
        
        # Ensure critical fields exist
        if 'title' not in clean_job or not clean_job['title'] or clean_job['title'] == "not mentioned":
            clean_job['title'] = "Untitled Position"
        
        if 'company' not in clean_job or not clean_job['company'] or clean_job['company'] == "not mentioned":
            clean_job['company'] = "Unknown Company"
        
        return clean_job

    async def _extract_jobs_with_llm_context(self, markdown_content: str, source_url: str = "") -> List[Dict[str, Any]]:
        """Extract job listings from markdown content with proper LLM context."""
        if not markdown_content or not markdown_content.strip():
            logger.warning("Empty or whitespace-only markdown content received")
            return []
        
        logger.info(f"🎯 Starting LLM job extraction from {len(markdown_content)} characters of markdown")
        
        # Handle large content by chunking
        if len(markdown_content) > 6000:
            logger.info("Large markdown content detected, using chunked extraction")
            return await self._extract_jobs_from_chunks(markdown_content)
        
        # Enhanced prompt with better context
        context_info = f"Source URL: {source_url}" if source_url else "Source: Job listing page"
        
        prompt = f"""You are analyzing a job listing page converted to markdown format. Extract ALL job positions mentioned.

CONTEXT: {context_info}

TASK: Extract every job listing, position, role, or career opportunity from this markdown content.

OUTPUT FORMAT - Return ONLY a valid JSON array:
[
  {{
    "title": "Exact job title or position name",
    "company": "Company name if mentioned, otherwise 'not mentioned'",
    "location": "Work location if specified, otherwise 'not mentioned'",
    "description": "Brief job description if available, otherwise 'not mentioned'",
    "salary": "Salary/compensation if mentioned, otherwise 'not mentioned'",
    "requirements": "Key requirements or qualifications, otherwise 'not mentioned'",
    "contrat-type": "Employment type (Full-time, Part-time, etc.), otherwise 'not mentioned'",
    "required_skill": "Required skills if mentioned, otherwise 'not mentioned'",
    "post_date": "Job posting date if available, otherwise 'not mentioned'"
  }}
]

IMPORTANT RULES:
- Extract ALL distinct job positions - don't limit the number
- Use "not mentioned" for missing information, never null or empty strings
- Be generous in identifying jobs - include any position, role, or opportunity
- If you see the same job multiple times, include it only once
- Focus on actual job openings, not just company information

MARKDOWN CONTENT TO ANALYZE:
{markdown_content}

JSON OUTPUT:"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a job extraction specialist. Analyze markdown content and extract job listings in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False
        }

        try:
            # Simple retry with backoff for rate limits
            for attempt in range(3):
                response = await self.http_client.post(self.groq_api_url, json=payload)
                if response.status_code == 429:
                    wait_s = 1 * (2 ** attempt)
                    logger.warning(f"LLM rate limited (429). Retrying in {wait_s}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait_s)
                    continue
                break
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                
                logger.info(f"📝 LLM response length: {len(content)} characters")
                if len(content) > 0 and len(content) < 500:
                    logger.info(f"📝 Full LLM response: {content}")
                elif len(content) >= 500:
                    logger.info(f"📝 LLM response preview: {content[:500]}...")
                
                # Extract and parse jobs
                jobs = self._extract_jobs_json(content)
                
                if not jobs:
                    logger.info("No jobs found with primary extraction, trying aggressive method")
                    jobs = await self._aggressive_job_extraction(markdown_content)
                
                # Clean and validate job data
                cleaned_jobs = []
                for job in jobs:
                    clean_job = self._clean_job_data(job)
                    if clean_job.get('title') and clean_job['title'] != "not mentioned":
                        cleaned_jobs.append(clean_job)
                
                logger.info(f"🎯 Successfully extracted {len(cleaned_jobs)} valid jobs")
                return cleaned_jobs
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error in LLM job extraction: {e}", exc_info=True)
            return []

    async def _extract_jobs_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract job listings directly from HTML content using LLM."""
        if not html_content.strip():
            return []
        
        logger.info(f"🎯 Starting job extraction from {len(html_content)} characters of HTML")

        # For large content, truncate to avoid token limits
        if len(html_content) > 8000:
            html_content = html_content[:8000]
            logger.info("HTML content truncated to 8000 characters")

        # Early exit to avoid unnecessary LLM calls
        if len(html_content) < 200 and not self._has_job_signals(html_content):
            logger.info("Skipping HTML LLM extraction: content too short and no job signals")
            return []
        
        # Enhanced extraction prompt for HTML content
        prompt = f"""Extract ALL job listings from this HTML content. Return ONLY a JSON array.

INSTRUCTIONS:
- Look for job titles, positions, roles, or career opportunities
- Extract company information, job descriptions, requirements, and other details
- Be generous in what you consider a job listing
- Return structured JSON with consistent fields

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{
    "title": "Job Title",
    "company": "Company Name or 'not mentioned'",
    "location": "Work location or 'not mentioned'",
    "description": "Brief job description or 'not mentioned'",
    "salary": "Salary information or 'not mentioned'",
    "requirements": "Key qualifications or 'not mentioned'",
    "contrat-type": "Full-time, Part-time, etc. or 'not mentioned'",
    "required_skill": "Key skills needed or 'not mentioned'",
    "post_date": "When job was posted or 'not mentioned'"
  }}
]

HTML CONTENT:
{html_content}

JSON OUTPUT:"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a job extraction AI. Extract job listings from HTML and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False
        }

        try:
            response = await self.http_client.post(self.groq_api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                
                logger.info(f"📝 Raw LLM extraction output: {len(content)} characters")
                
                # Try multiple extraction strategies
                jobs = self._extract_jobs_json(content)
                
                if not jobs:
                    logger.info("No jobs found with standard extraction, trying aggressive method")
                    jobs = await self._aggressive_job_extraction_html(html_content)
                
                # Clean and enhance job data
                cleaned_jobs = []
                for job in jobs:
                    clean_job = self._clean_job_data(job)
                    cleaned_jobs.append(clean_job)
                
                logger.info(f"🎯 Successfully extracted {len(cleaned_jobs)} jobs from HTML")
                return cleaned_jobs
            else:
                logger.error(f"Failed to extract jobs from HTML: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error extracting jobs from HTML: {e}")
            return []

    async def _aggressive_job_extraction_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Aggressively extract job titles from HTML content."""
        import re
        
        logger.info("Using aggressive HTML job extraction method")
        
        # Simple prompt focusing on job titles only
        aggressive_prompt = f"""Look through this HTML and find ANY job titles or positions mentioned.
Be extremely generous - include anything that might be a job.
Return ONLY a JSON array with job titles.

Example:
[
  {{"title": "Software Engineer"}},
  {{"title": "Marketing Manager"}}
]

HTML Content (first 4000 chars):
{html_content[:4000]}

JSON OUTPUT:"""
    
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Extract job titles from HTML. Return only JSON."},
                {"role": "user", "content": aggressive_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            response = await self.http_client.post(self.groq_api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '[]')
                
                job_objects = self._extract_jobs_json(content)
                
                # If still no jobs, try regex on HTML
                if not job_objects:
                    # Look for common job-related patterns in HTML
                    patterns = [
                        r'<title[^>]*>([^<]*(?:job|career|position|role)[^<]*)</title>',
                        r'<h[1-6][^>]*>([^<]*(?:job|career|position|role)[^<]*)</h[1-6]>',
                        r'job[_-]?title["\s]*[:=]["\s]*([^"<>\n]{3,50})',
                        r'position["\s]*[:=]["\s]*([^"<>\n]{3,50})'
                    ]
                    
                    titles = []
                    for pattern in patterns:
                        matches = re.findall(pattern, html_content, re.IGNORECASE)
                        titles.extend([match.strip() for match in matches if len(match.strip()) > 3])
                    
                    job_objects = [{"title": title} for title in set(titles)[:5]]  # Limit to 5 unique titles
                
                # Add default fields
                enhanced_jobs = []
                for job in job_objects:
                    job_info = {
                        "title": job.get("title", "Unknown Position"),
                        "company": "not mentioned",
                        "location": "not mentioned", 
                        "description": "not mentioned",
                        "salary": "not mentioned",
                        "requirements": "not mentioned",
                        "contrat-type": "not mentioned",
                        "required_skill": "not mentioned",
                        "post_date": "not mentioned"
                    }
                    enhanced_jobs.append(job_info)
                    
                return enhanced_jobs
            else:
                logger.error(f"Aggressive HTML extraction LLM API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Aggressive HTML extraction failed: {e}")
            return []

    async def _stream_groq_response(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Helper method to handle streaming and parsing of Groq API responses.
        
        This method buffers the response to maintain markdown formatting and streams
        complete lines or logical chunks to ensure proper rendering in the frontend.
        """
        try:
            async with self.http_client.stream("POST", self.groq_api_url, json=payload) as response:
                if response.status_code != 200:
                    # Avoid accessing response.text before reading
                    status = response.status_code
                    msg = "rate limited" if status == 429 else "error"
                    error_msg = f"LLM API {msg}: {status}"
                    logger.error(error_msg)
                    yield json.dumps({"type": "error", "message": error_msg})
                    return

                buffer = ""
                in_code_block = False
                code_block_delimiter = ""
                
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):
                        continue
                    
                    data = line[6:].strip()
                    if data == '[DONE]':
                        break
                        
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                buffer += content
                                
                                # Handle code block delimiters
                                if '```' in content:
                                    if in_code_block:
                                        in_code_block = False
                                        code_block_delimiter = ""
                                    else:
                                        in_code_block = True
                                        # Get the language if specified (e.g., ```python)
                                        parts = content.split('```')
                                        if len(parts) > 1 and parts[1].strip():
                                            code_block_delimiter = '```' + parts[1] + '\n'
                                # Stream content more aggressively to ensure response reaches frontend
                                if '\n' in buffer or len(buffer) > 100:
                                    # Helper: detect bracketed phase lines and emit as thinking_step
                                    def _emit_line(l: str):
                                        s = l.strip()
                                        if not s:
                                            return False
                                        import re
                                        m = re.match(r"^\[(initial analysis|planning|tool selection|execution|observation|reflection|final synthesis)\]\s*(.*)$", s, re.IGNORECASE)
                                        if m:
                                            phase_raw = m.group(1).lower().replace(' ', '_')
                                            content_rest = m.group(2) or ''
                                            evt = {"type": "thinking_step", "phase": phase_raw, "content": content_rest}
                                            yield json.dumps(evt) + "\n"
                                            return True
                                        return False

                                    if '\n' in buffer:
                                        lines = buffer.split('\n')
                                        for l in lines[:-1]:
                                            # Try emit as thinking_step; if not, emit as normal response
                                            emitted = False
                                            for out in _emit_line(l):
                                                emitted = True
                                                yield out
                                            if not emitted and l.strip():
                                                yield json.dumps({"type": "response", "content": l + '\n'}) + "\n"
                                        buffer = lines[-1]
                                    else:
                                        # Single long chunk without newline
                                        emitted = False
                                        for out in _emit_line(buffer):
                                            emitted = True
                                            yield out
                                        if not emitted:
                                            yield json.dumps({"type": "response", "content": buffer}) + "\n"
                                        buffer = ""
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse LLM chunk: {e}")
                
                # Yield any remaining content in the buffer
                if buffer.strip():
                    # Final pass: try to emit as thinking_step first
                    s = buffer.strip('\n')
                    import re
                    m = re.match(r"^\[(initial analysis|planning|tool selection|execution|observation|reflection|final synthesis)\]\s*(.*)$", s, re.IGNORECASE)
                    if m:
                        phase_raw = m.group(1).lower().replace(' ', '_')
                        content_rest = m.group(2) or ''
                        yield json.dumps({"type": "thinking_step", "phase": phase_raw, "content": content_rest}) + "\n"
                    else:
                        yield json.dumps({"type": "response", "content": buffer}) + "\n"
                    
        except Exception as e:
            error_msg = f"Error streaming Groq response: {e}"
            logger.error(error_msg)
            yield json.dumps({"type": "error", "message": error_msg}) + "\n"

    async def _analyze_user_intent(self, user_message: str, has_cv_file: bool) -> Dict[str, Any]:
        """
        Analyze user intent using LLM, supporting job search by company name or direct job/career URL.
        Returns a dictionary with intent analysis results.
{{ ... }}
        """
        prompt = f"""You are a powerful intent analysis engine. Your task is to analyze the user's message and return a structured JSON object.

    **User Message:** "{user_message}"
    **CV Attached:** {has_cv_file}

    **Instructions:**
    1.  **Determine Primary Intent:** Choose ONE from 'comparison', 'cv_analysis', 'job_search', 'general_conversation'.
        - **comparison**: The user wants to compare their CV to jobs. This requires a job source (company name or URL).
        - **cv_analysis**: The user wants their CV analyzed (e.g., "summarize my CV").
        - **job_search**: The user wants to find jobs without using a CV.
    2.  **Extract Entities (CRITICAL):**
        - **company_names**: Find all company names mentioned. Example: "Google", "Microsoft".
        - **urls**: Find all complete URLs. Look for text starting with 'http://' or 'https://'. This is extremely important. Example: "https://www.keejob.com/offres-emploi/". If a URL is present, you MUST extract it.
    3.  **Summarize Request:** Provide a one-sentence summary of the user's core request.

    **Output Format:**
    You MUST respond with ONLY a valid JSON object. Do not add any other text.
    {{
        "primary_intent": "...",
        "company_names": ["..."],
        "urls": ["..."],
        "summary": "..."
    }}
    """
        payload = {
            "model": self.model_name, 
            "messages": [{"role": "user", "content": prompt}], 
            "max_tokens": 2048, 
            "temperature": 0.0
        }
        
        try:
            response = await self.http_client.post(self.groq_api_url, json=payload)
            if response.status_code != 200:
                logger.error(f"LLM intent analysis failed: {response.text}")
                return {
                    "primary_intent": "general_conversation",
                    "company_names": [],
                    "urls": [],
                    "summary": "I'm having trouble understanding your request. Could you please rephrase?"
                }
                
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Try to parse the JSON response
            try:
                # Find the JSON part in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    intent_data = json.loads(content[json_start:json_end])
                    
                    # Ensure all required fields are present
                    intent_data.setdefault("primary_intent", "general_conversation")
                    intent_data.setdefault("company_names", [])
                    intent_data.setdefault("urls", [])
                    intent_data.setdefault("summary", "")
                    
                    # Map primary_intent to workflow
                    workflow_map = {
                        "comparison": "comparison",
                        "cv_analysis": "cv_analysis",
                        "job_search": "job_search",
                        "general_conversation": "general"
                    }
                    intent_data["workflow"] = workflow_map.get(
                        intent_data["primary_intent"], 
                        "general"
                    )
                    
                    return intent_data
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode JSON from LLM response: {content}")
                logger.warning(f"JSON decode error: {e}")
                
        except Exception as e:
            logger.error(f"Error during intent analysis: {e}")
            
        # Default fallback response
        return {
            "primary_intent": "general_conversation",
            "company_names": [],
            "urls": [],
            "summary": "I'm having trouble understanding your request. Could you please rephrase?",
            "workflow": "general"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all components"""
        try:
            health_status = {
                "unified_host": False,
                "groq_api": False,
                "cv_client": False,
                "job_client": False,
                "initialized": self.initialized,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if not self.initialized:
                health_status["error"] = "Unified LLM Host not initialized"
                return health_status
            
            # Check Groq API
            try:
                await self._check_groq_health()
                health_status["groq_api"] = True
            except Exception as e:
                health_status["groq_error"] = str(e)
            
            # Check CV client
            try:
                cv_client = await self.get_cv_client()
                if cv_client and hasattr(cv_client, 'ping'):
                    cv_ping = await cv_client.ping()
                    health_status["cv_client"] = cv_ping.get("success", False)
                else:
                    health_status["cv_client"] = False
                health_status["cv_mcp_client"] = {"status": "initialized" if cv_client else "not_initialized"}
            except Exception as e:
                health_status["cv_error"] = str(e)
            
            # Check Job client
            try:
                job_client = await self.get_job_client()
                if job_client and hasattr(job_client, 'ping'):
                    job_ping = await job_client.ping()
                    health_status["job_client"] = job_ping.get("success", False)
                else:
                    health_status["job_client"] = False
                health_status["job_mcp_client"] = {"status": "initialized" if job_client else "not_initialized"}
            except Exception as e:
                health_status["job_error"] = str(e)
            
            # Overall health
            health_status["unified_host"] = all([
                health_status["groq_api"],
                health_status["cv_client"],
                health_status["job_client"],
                self.initialized
            ])
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "unified_host": False,
                "error": str(e),
                "initialized": self.initialized,
                "timestamp": asyncio.get_event_loop().time()
            }

    async def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        return {
            "model_name": self.model_name,
            "groq_api_url": self.groq_api_url,
            "timeout": self.timeout,
            "cv_server_script": self.cv_mcp_server_script,
            "job_server_script": self.job_mcp_server_script,
            "initialized": self.initialized,
            "version": "1.0.0",
            "capabilities": [
                "cv_processing",
                "job_search", 
                "cv_job_matching",
                "general_conversation"
            ],
            "intent_analysis": {
                "method": "llm_only",
                "fallback": False,
                "reasoning_required": True
            }
        }

    async def _safe_error_text(self, response: Optional[httpx.Response]) -> str:
        """Safely read and return response text for logging in error paths.

        Handles cases where the response is a streaming response that hasn't been read yet.
        """
        if response is None:
            return "<no response>"
        try:
            # Attempt to read the body in case it wasn't consumed (e.g., streaming)
            try:
                await response.aread()
            except Exception:
                pass
            # Now accessing .text should not raise ResponseNotRead
            return response.text
        except Exception:
            try:
                return f"HTTP {response.status_code} {response.reason_phrase or ''}".strip()
            except Exception:
                return "<unreadable response>"

    async def _generate_general_response(self, user_message: str, intent: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate a general response for a given user message and intent"""
        system_prompt = f"""You are a helpful career and CV assistant. You have analyzed the user's intent and determined this is a general conversation.

**INTENT ANALYSIS CONTEXT:**
- LLM Reasoning: {intent.get('reasoning', 'Not provided')}
- Confidence: {intent.get('confidence', 0.0)}
- Primary Intent: {intent.get('primary_intent', 'general_conversation')}

**Your Capabilities & Routing:**
1. **CV Analysis & Improvement**: UnifiedHost → CVMCPClient → CVMCPServer
   - Analyzing CVs, providing feedback, extracting information, and suggesting improvements.
2. **Job Search Assistance**: UnifiedHost → JobMCPClient → JobMCPServer
   - Finding job opportunities at companies, career pages, and application guidance.
3. **CV-Job Matching**: UnifiedHost → CVMCPClient & JobMCPClient → CVMCPServer & JobMCPServer
   - Comparing a CV against job descriptions to assess suitability.

**Your Instructions:**
1. Acknowledge the user's message in a friendly, conversational tone.
2. Based on the primary intent, gently guide the user toward one of your core functions if appropriate. For example, if they say "thanks," you can reply with "You're welcome! Is there anything else I can help you with, like analyzing a CV or searching for jobs?"
3. Keep your response concise and helpful.

Start your response naturally without explicitly mentioning the intent analysis."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": True
        }

        try:
            async with self.http_client.stream("POST", self.groq_api_url, json=payload) as response:
                response.raise_for_status()
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):
                        continue
                    data_str = line[len('data: '):].strip()
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            buffer += content
                            if '\n' in buffer:
                                lines = buffer.split('\n')
                                for l in lines[:-1]:
                                    if l.strip():
                                        yield json.dumps({"type": "response", "content": l + '\n'}) + "\n"
                                buffer = lines[-1]
                            elif len(buffer) > 200:
                                yield json.dumps({"type": "response", "content": buffer}) + "\n"
                                buffer = ""
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from stream: {data_str}")
                if buffer.strip():
                    yield json.dumps({"type": "response", "content": buffer}) + "\n"
        except httpx.HTTPStatusError as e:
            error_text = await self._safe_error_text(e.response)
            logger.error(f"HTTP error from Groq API: {e.response.status_code} - {error_text}")
            yield json.dumps({"type": "error", "message": "Error: I'm having trouble connecting to my core systems right now. Please try again later."}) + "\n"
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            yield json.dumps({"type": "error", "message": "Error: I encountered an unexpected issue while processing your request."}) + "\n"

# Interactive CLI mode
async def interactive_mode():
    """Interactive CLI mode for testing the Unified LLM Host"""
    
    print("🚀 Unified LLM Host - Interactive Mode")
    print("=" * 50)
    
    # Initialize the host
    host = UnifiedLLMHost()
    
    try:
        # Initialize all components
        print("\n🔄 Initializing system...")
        await host.initialize()
        print("✅ System initialized successfully!")
        
        # Show system info
        info = await host.get_system_info()
        print(f"\n📊 System Info:")
        print(f"   Model: {info['model_name']}")
        print(f"   Capabilities: {', '.join(info['capabilities'])}")
        print(f"   Intent Analysis: {info['intent_analysis']['method']}")
        
        # Health check
        health = await host.health_check()
        if health['unified_host']:
            print("✅ All systems healthy")
        else:
            print("⚠️  Some components may have issues")
            for component, status in health.items():
                if component.endswith('_error'):
                    print(f"   Error in {component}: {status}")
        
        print("\n" + "=" * 50)
        print("💬 Interactive Chat Mode")
        print("Commands:")
        print("  /file <path>  - Set CV file path")
        print("  /clear        - Clear CV file path") 
        print("  /health       - Check system health")
        print("  /info         - Show system info")
        print("  /debug        - Toggle debug mode")
        print("  /help         - Show this help")
        print("  /quit         - Exit interactive mode")
        print("=" * 50)
        
        # Interactive session state
        session_id = f"interactive_{int(asyncio.get_event_loop().time())}"
        current_file_path = ""
        debug_mode = False
        
        print(f"\n🔧 Session ID: {session_id}")
        print(f"💡 Type your message or use commands. Ready to help!")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input.split(' ', 1)
                    command = command_parts[0].lower()
                    
                    if command == '/quit':
                        print("👋 Goodbye!")
                        break
                    
                    elif command == '/file':
                        if len(command_parts) > 1:
                            new_path = command_parts[1].strip()
                            if Path(new_path).exists():
                                current_file_path = new_path
                                print(f"✅ CV file set: {current_file_path}")
                            else:
                                print(f"❌ File not found: {new_path}")
                        else:
                            print("Usage: /file <path_to_cv_file>")
                    
                    elif command == '/clear':
                        current_file_path = ""
                        print("✅ CV file path cleared")
                    
                    elif command == '/health':
                        print("🔍 Checking system health...")
                        health = await host.health_check()
                        print(json.dumps(health, indent=2))
                    
                    elif command == '/info':
                        info = await host.get_system_info()
                        print(json.dumps(info, indent=2))
                    
                    elif command == '/debug':
                        debug_mode = not debug_mode
                        print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                    
                    elif command == '/help':
                        print("\nCommands:")
                        print("  /file <path>  - Set CV file path")
                        print("  /clear        - Clear CV file path") 
                        print("  /health       - Check system health")
                        print("  /info         - Show system info")
                        print("  /debug        - Toggle debug mode")
                        print("  /help         - Show this help")
                        print("  /quit         - Exit interactive mode")
                    
                    else:
                        print(f"❌ Unknown command: {command}")
                    
                    continue
                
                # Process user message
                print("🤖 Processing...")
                
                request_data = {
                    "message": user_input,
                    "file_path": current_file_path,
                    "session_id": session_id
                }
                
                # Show request details in debug mode
                if debug_mode:
                    print(f"\n🐛 Debug - Request:")
                    print(f"   Message: {user_input}")
                    print(f"   File: {current_file_path or 'None'}")
                    print(f"   Session: {session_id}")
                
                # Process the request
                start_time = asyncio.get_event_loop().time()
                result = await host.process_unified_request(request_data)
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Show response
                print(f"\n🤖 Assistant ({result.get('intent', 'unknown')}):")
                print(f"{result.get('response', 'No response generated')}")
                
                # Show processing info in debug mode
                if debug_mode and 'processing_info' in result:
                    processing_info = result['processing_info']
                    print(f"\n🐛 Debug - Processing Info:")
                    print(f"   Success: {result.get('success', False)}")
                    print(f"   Intent: {result.get('intent', 'unknown')}")
                    print(f"   Processing Time: {processing_time:.2f}s")
                    print(f"   Confidence: {processing_info.get('confidence', 0.0):.2f}")
                    print(f"   Primary Intent: {processing_info.get('primary_intent', 'unknown')}")
                    
                    reasoning = processing_info.get('llm_reasoning', 'No reasoning provided')
                    if len(reasoning) > 100:
                        reasoning = reasoning[:100] + "..."
                    print(f"   LLM Reasoning: {reasoning}")
                    
                    if 'error_details' in processing_info:
                        print(f"   Error: {processing_info['error_details']}")
                
                # Show file status
                if current_file_path:
                    print(f"\n📄 Current CV: {Path(current_file_path).name}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if debug_mode:
                    import traceback
                    print(f"🐛 Debug - Traceback:")
                    traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        
    finally:
        print("\n🔄 Shutting down...")
        await host.shutdown()
        print("✅ Shutdown complete")

# Example usage and testing
async def batch_test():
    """Batch testing mode with predefined test cases"""
    
    print("🧪 Unified LLM Host - Batch Test Mode")
    print("=" * 50)
    
    # Initialize the host
    host = UnifiedLLMHost()
    
    try:
        # Initialize all components
        await host.initialize()
        
        # Test health check
        health = await host.health_check()
        print("Health Check:", json.dumps(health, indent=2))
        
        # Test system info
        info = await host.get_system_info()
        print("System Info:", json.dumps(info, indent=2))
        
        # Example request processing
        test_requests = [
            {
                "message": "Can you analyze my CV and tell me about my skills?",
                "file_path": "/path/to/cv.pdf",
                "session_id": "test_session_1"
            },
            {
                "message": "Find job opportunities at Google and Microsoft",
                "file_path": "",
                "session_id": "test_session_2"
            },
            {
                "message": "How does my CV match with Amazon jobs?",
                "file_path": "/path/to/cv.pdf", 
                "session_id": "test_session_3"
            },
            {
                "message": "Hello, what can you help me with?",
                "file_path": "",
                "session_id": "test_session_4"
            }
        ]
        
        # Process test requests
        for i, request in enumerate(test_requests):
            print(f"\n=== Test Request {i+1} ===")
            print(f"Message: {request['message']}")
            
            result = await host.process_unified_request(request)
            print(f"Intent: {result.get('intent', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            
            if 'processing_info' in result:
                reasoning = result['processing_info'].get('llm_reasoning', 'No reasoning provided')
                confidence = result['processing_info'].get('confidence', 0.0)
                print(f"LLM Reasoning: {reasoning}")
                print(f"Confidence: {confidence}")
            
            print(f"Response: {result.get('response', 'No response')[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Cleanup
        await host.shutdown()

def main():
    """Main entry point - choose between interactive and batch mode"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ['interactive', 'i', 'chat']:
            asyncio.run(interactive_mode())
        elif mode in ['batch', 'b', 'test']:
            asyncio.run(batch_test())
        elif mode in ['help', 'h', '--help']:
            print("Unified LLM Host - Usage:")
            print("  python orchestrator.py interactive  - Start interactive chat mode")
            print("  python orchestrator.py batch       - Run batch test mode")
            print("  python orchestrator.py help        - Show this help")
        else:
            print(f"Unknown mode: {mode}")
            print("Use 'python orchestrator.py help' for usage information")
    else:
        # Default to interactive mode
        print("Starting interactive mode (use 'python orchestrator.py help' for options)")
        asyncio.run(interactive_mode())

if __name__ == "__main__":
    main()