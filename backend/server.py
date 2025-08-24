import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Set, List, Optional
import httpx
from urllib.parse import urlparse, urljoin, urlunparse
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import re
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import logging 
import mcp.types
from urllib.robotparser import RobotFileParser
import asyncio
from collections import deque

# Force UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Configuration du navigateur pour crawl4ai
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    extra_args=[
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    ]
)

# MCP server instance
mcp = FastMCP(name="JobOfferServer")

def normalize_url(url: str, base_url: str = None) -> str:
    """Normalize URL by removing fragments and query parameters, handling relative URLs"""
    if base_url and not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)
    
    parsed = urlparse(url)
    # Remove fragment and normalize
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path.rstrip('/') or '/',
        parsed.params,
        parsed.query,
        ''  # Remove fragment
    ))
    return normalized

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain"""
    domain1 = urlparse(url1).netloc.lower()
    domain2 = urlparse(url2).netloc.lower()
    return domain1 == domain2

def extract_links_from_html(html: str, base_url: str) -> List[str]:
    """Extract all internal links from HTML content"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for tag in soup.find_all(['a', 'link'], href=True):
            href = tag['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
                
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            normalized = normalize_url(absolute_url)
            
            # Only include same-domain links
            if is_same_domain(normalized, base_url):
                links.append(normalized)
        
        return list(set(links))  # Remove duplicates
    except Exception as e:
        print(f"❌ Error extracting links: {e}", file=sys.stderr)
        return []

def extract_job_links_from_html(html: str, base_url: str) -> List[str]:
    """Extract likely job posting links by analyzing surrounding context (classes/ids/text)."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        job_links: List[str] = []
        ctx_pattern = re.compile(r"job|offer|offre|career|recruit|recrut|opening|position|vacanc|emploi", re.IGNORECASE)

        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            # textual hints on the anchor itself
            text = (a.get_text() or '').strip().lower()
            if any(t in text for t in ['apply', 'postuler', 'offre', 'offer', 'emploi', 'job', 'career', 'details']):
                abs_url = normalize_url(urljoin(base_url, href))
                if is_same_domain(abs_url, base_url) and not should_skip_url(abs_url):
                    job_links.append(abs_url)
                    continue

            # structural hints: parent containers with job-related classes/ids
            parent = a
            for _ in range(3):  # look up to 3 levels up
                parent = parent.parent
                if not parent:
                    break
                id_class = ' '.join(filter(None, [parent.get('id', ''), ' '.join(parent.get('class', []) or [])]))
                if ctx_pattern.search(id_class or ''):
                    abs_url = normalize_url(urljoin(base_url, href))
                    if is_same_domain(abs_url, base_url) and not should_skip_url(abs_url):
                        job_links.append(abs_url)
                        break

        # Pagination next/previous links
        for rel in ['next']:
            for link in soup.find_all('a', attrs={'rel': rel}, href=True):
                abs_url = normalize_url(urljoin(base_url, link['href']))
                if is_same_domain(abs_url, base_url) and not should_skip_url(abs_url):
                    job_links.append(abs_url)

        return list(dict.fromkeys(job_links))  # dedupe, keep order
    except Exception as e:
        print(f"❌ Error extracting job links: {e}", file=sys.stderr)
        return []

def is_job_related_url(url: str) -> bool:
    """Heuristic to check if URL might be job-related based on path and common patterns"""
    url_lower = url.lower()
    patterns = [
        r"/(jobs?|careers?)\b",
        r"/(emploi|offres?|recrutement)\b",
        r"/(opportunit\w+|opportunities)\b",
        r"/(positions?|vacanc\w+)\b",
        r"/(apply|postuler)\b",
        r"/(join(-|_)us)\b",
        r"/(opening|openings)\b",
        r"/(offer|offre)[-_/]",
        r"/(job|position)[-_/][0-9A-Za-z]",
    ]
    return any(re.search(p, url_lower) for p in patterns)

def is_job_detail_url(url: str) -> bool:
    """Detect likely job detail pages universally (id + slug, offer pages, etc.)."""
    url_lower = url.lower()
    detail_patterns = [
        # Common listing base then numeric id segment
        r"/(offres-emploi|jobs?|careers?|emploi|recrutement)/\d{3,}(/|$)",
        # job or offer with id in path
        r"/(job|offer|offre)[-_/]?\d{2,}(/|-)",
        # path ends with long slug typically for a single posting
        r"/(jobs?|offres?-emploi|career|recrutement)/[0-9a-z\-]{10,}(/)?$",
        # presence of typical apply endpoints deeper in path
        r"/(apply|postuler)(/|$)",
    ]
    return any(re.search(p, url_lower) for p in detail_patterns)

def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped (file downloads, external resources, etc.)"""
    url_lower = url.lower()
    skip_patterns = [
        r'\.(pdf|doc|docx|xls|xlsx|zip|rar|exe|dmg)$',
        r'/download[s]?/',
        r'/api/',
        r'/admin/',
        r'/wp-admin/',
        r'/wp-content/',
        r'/assets/',
        r'/static/',
        r'/css/',
        r'/js/',
        r'/images?/',
        r'/img/',
        r'/media/',
        r'/files?/'
    ]
    return any(re.search(pattern, url_lower) for pattern in skip_patterns)

#This is commonly used in web crawlers to respect site policies and avoid legal or ethical issues.
def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt is missing or can't be read, assume allowed
        return True

def send_progress_notification(message: str, percentage: float = None, title: str = "Job HTML Finder Progress"):
    """Send progress notification - disabled to avoid validation errors"""
    # Temporarily disabled to prevent MCP validation errors
    # The orchestrator already provides comprehensive logging
    pass

async def quick_job_content_check(html: str, url: str) -> bool:
    """
    Quick check to determine if HTML content is job-related.
    This is a lightweight version of the precheck function.
    """
    try:
        # URL-based check first
        if is_job_related_url(url):
            return True
        
        # Content-based patterns (simplified)
        job_patterns = [
            r'\b(apply|candidat|position|emploi|job|career|recrutement|opportunity)\b',
            r'<form[^>]*action=["\'][^"\']*(apply|submit|candidat)[^"\']*["\']',
            r'\b(full[- ]?time|part[- ]?time|stage|internship|cdi|cdd)\b',
            r'(salary|salaire|benefits|remote|télétravail)',
            r'(experience|qualification|requirement|compétence)'
        ]
        
        html_lower = html.lower()[:5000]  # Check only first 5000 chars for speed
        matches = sum(1 for pattern in job_patterns if re.search(pattern, html_lower, re.IGNORECASE))
        
        return matches >= 2
    
    except Exception:
        return False

# @mcp.tool(annotations={"title": "get_official_website_and_generate_job_urls"})
async def get_official_website_and_generate_job_urls(company_name: str) -> dict:
    """
    Recherche le site officiel d'une société via DuckDuckGo et génère des URLs potentielles d'offres à partir de ce site.
    """
    try:
        send_progress_notification(f"Recherche du site officiel de {company_name}...", 10)
        
        def get_official_website(company_name: str) -> str:
            query = f"{company_name} site officiel"
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=5):
                        url = r.get("href") or r.get("url") or ""
                        # Allow linkedin.com as an official site (e.g., for the LinkedIn company itself),
                        # but still exclude Wikipedia.
                        if url and "wikipedia.org" not in url:
                            clean = (url.split("?")[0]).strip()
                            parsed = urlparse(clean)
                            if parsed.scheme and parsed.netloc:
                                # Normalize to origin to avoid inheriting deep paths like /download
                                return f"{parsed.scheme}://{parsed.netloc}"
                            return clean
                return ""
            except Exception as e:
                send_progress_notification(f"Erreur DuckDuckGo: {str(e)}", 0)
                return ""

        def generate_job_urls(base_url: str, company_name: str) -> list:
            # Ensure base is the site origin
            parsed = urlparse(base_url)
            base_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/") if parsed.scheme and parsed.netloc else base_url.rstrip("/")
            patterns = [
                "/jobs",
                "/careers",
                "/career",
                "/recrutement",
                "/offres-emploi",
                "/job-offer",
                "/join-us",
                "/about/careers"
            ]
            urls = [f"{base_origin}{p}" for p in patterns]
            # Include homepage as a fallback (some sites link careers from home)
            urls.append(base_origin)
            # Deduplicate
            return list(dict.fromkeys(urls))

        site = get_official_website(company_name)
        if not site:
            return {"success": False, "company_name": company_name, "reason": "Site officiel introuvable."}
        
        send_progress_notification(f"Site trouvé: {site}", 50)
        urls = generate_job_urls(site, company_name)
        send_progress_notification(f"Génération de {len(urls)} URLs potentielles", 100)
        
        return {"success": True, "company_name": company_name, "official_website": site, "possible_job_urls": urls}
    
    except Exception as e:
        return {"success": False, "company_name": company_name, "reason": f"Erreur: {str(e)}"}

# @mcp.tool(annotations={"title": "precheck_job_offer_url"})
async def precheck_job_offer_url(url: str) -> dict:
    """
    Précheck d'une URL (1000 premiers caractères, motifs d'offres détectés).
    """
    OFFER_PATTERNS = [
        r'/(jobs|careers|recrutement|opportunit\w+|emploi)/?',
        r'apply(-now)?', r'\b(job|emploi)\b', r'current openings',
        r'<a[^>]+href=["\'].*(job|emploi|offre|career|apply).*["\']',
        r'<form[^>]*action=["\'][^"\']*(apply|submit)[^"\']*["\']',
        r'<meta[^>]+(job|career|recrutement)[^>]*>',
        r'espace[\s_-]?candidat[s]?',
    ]
    
    # More specific error patterns - focusing on clear error indicators
    ERROR_PATTERNS = [
        r'<title[^>]*>\s*404\s*</title>',
        r'<title[^>]*>.*?(page not found|page introuvable).*?</title>',
        r'<h1[^>]*>\s*404\s*</h1>',
        r'<div[^>]*class="[^"]*error-page[^"]*"[^>]*>',
        r'<div[^>]*class="[^"]*page-404[^"]*"[^>]*>',
        r'<div[^>]*id="[^"]*error-page[^"]*"[^>]*>',
        r'<div[^>]*id="[^"]*page-404[^"]*"[^>]*>'
    ]
    
    # Specific phrases that definitely indicate empty job listings
    EMPTY_JOB_PHRASES = [
        r'<div[^>]*>\s*aucune offre (d\'emploi)? (n\'est)? disponible\s*</div>',
        r'<p[^>]*>\s*aucune offre (d\'emploi)? (n\'est)? disponible\s*</p>',
        r'<div[^>]*>\s*no (job )?positions (are )?available\s*</div>',
        r'<p[^>]*>\s*no (job )?positions (are )?available\s*</p>'
    ]
    
    try:
        send_progress_notification(f"Vérification de {url}...", 10)

        # Quick URL heuristic to short-circuit obvious job pages
        url_lower = url.lower()
        parsed = urlparse(url)
        # Known job board/provider hosts or paths
        strong_job_hosts = (
            'boards.greenhouse.io', 'greenhouse.io', 'jobs.lever.co', 'lever.co',
            'workable.com', 'smartrecruiters.com', 'ashbyhq.com', 'jobvite.com',
            'icims.com', 'myworkdayjobs.com', 'eightfold.ai', 'recruitee.com'
        )
        strong_job_paths = ('/jobs', '/careers', '/career', '/recrutement', '/emploi', '/opportunities')
        # Negative hosts/paths (common non-listing entry points)
        weak_hosts = ('linkedin.com', 'www.linkedin.com')
        # Blocklist of non-job content platforms
        blocked_hosts = (
            'zhihu.com', 'www.zhihu.com', 'medium.com', 'www.medium.com',
            'github.com', 'www.github.com', 'stackoverflow.com', 'www.stackoverflow.com',
            'facebook.com', 'www.facebook.com', 'twitter.com', 'www.twitter.com',
            'x.com', 'www.x.com', 'youtube.com', 'www.youtube.com',
            'bilibili.com', 'www.bilibili.com'
        )

        url_bonus = 0
        if parsed.scheme in ('http', 'https'):
            # Hard block unsupported content platforms
            if any(h == parsed.netloc for h in blocked_hosts):
                return {
                    "success": True,
                    "url": url,
                    "is_offer_page": False,
                    "reason": "Domain is not a job platform"
                }
            if any(host in parsed.netloc for host in strong_job_hosts):
                url_bonus += 2
            if any(path in parsed.path for path in strong_job_paths):
                url_bonus += 1
            # Penalize well-known generic hosts unless they have job subpaths
            if any(h == parsed.netloc for h in weak_hosts) and not any(p in parsed.path for p in ('/jobs', '/company/')):
                url_bonus -= 2

        if not is_allowed_by_robots(url):
            return {
                "success": False,
                "url": url,
                "is_offer_page": False,
                "reason": "Blocked by robots.txt"
            }

        # Lightweight HTTP fetch with short timeout
        try:
            headers = {
                "User-Agent": browser_config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            timeout = httpx.Timeout(8.0, connect=5.0)
            async with httpx.AsyncClient(follow_redirects=True, headers=headers, timeout=timeout) as client:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    return {
                        "success": False,
                        "url": url,
                        "is_offer_page": False,
                        "reason": f"HTTP Error {resp.status_code}"
                    }
                text = resp.text[:200000] if resp.text else ""
        except Exception as ex_http:
            # Fall back to crawler if lightweight fetch fails
            text = ""

        if text:
            # Title check for errors
            m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            title = (m.group(1).strip().lower() if m else "")
            if re.search(r"\b(404|error|not found|page introuvable)\b", title):
                return {"success": True, "url": url, "is_offer_page": False, "reason": f"Error indicated in title: '{title}'"}

            # Pattern checks
            for pattern in ERROR_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    return {"success": True, "url": url, "is_offer_page": False, "reason": "Error page structure detected"}
            for pattern in EMPTY_JOB_PHRASES:
                if re.search(pattern, text, re.IGNORECASE):
                    return {"success": True, "url": url, "is_offer_page": False, "reason": "Empty job listing page"}

            score = 0
            matched = []
            for pat in OFFER_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    score += 1
                    matched.append(pat)
            # Add URL-derived score
            if url_bonus > 0:
                score += url_bonus
                matched.append("URL signals")
            return {
                "success": True,
                "url": url,
                # Require at least moderate confidence: either strong host OR (url/path + on-page signals)
                "is_offer_page": (url_bonus >= 2) or (score >= 3),
                "patterns_found": score,
                "matched_patterns": matched[:5],
                "snippet": text[:200] + "..." if len(text) > 200 else text
            }

        # If we couldn't get lightweight content, fallback to crawler with minimal config
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    session_id="precheck",
                )
            )
            if not result.success or not getattr(result, 'html', None):
                return {"success": False, "url": url, "is_offer_page": False, "reason": "Crawler fetch failed"}
            text = result.html

            score = 0
            matched = []
            for pat in OFFER_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    score += 1
                    matched.append(pat)
            if any(term in url_lower for term in ['job', 'career', 'emploi', 'offre', 'recrutement']):
                score += 2
                matched.append("URL structure")
            return {
                "success": True,
                "url": url,
                "is_offer_page": score >= 2,
                "patterns_found": score,
                "matched_patterns": matched[:5],
                "snippet": text[:200] + "..." if len(text) > 200 else text
            }
    except Exception as ex:
        return {
            "success": False,
            "url": url,
            "is_offer_page": False,
            "reason": str(ex)
        }

@mcp.tool(annotations={"title": "deep_crawl_job_pages"})
async def deep_crawl_job_pages(
    start_url: str, 
    max_pages: int = 20,
    max_depth: int = 2,
    delay_between_requests: float = 1.0
) -> Dict[str, Any]:
    """
    Perform deep crawling of job-related pages starting from a given URL.
    
    Args:
        start_url: The initial URL to start crawling from
        max_pages: Maximum number of pages to crawl (default: 10)
        max_depth: Maximum depth to crawl (default: 2)
        delay_between_requests: Delay between requests in seconds (default: 1.0)
    
    Returns:
        Dictionary containing crawled pages and statistics
    """
    try:
        print(f"🚀 Starting deep crawl from: {start_url}", file=sys.stderr)
        send_progress_notification("Initializing deep crawl...", 0)
        
        # Initialize crawling state
        visited_urls: Set[str] = set()
        crawled_pages: List[Dict] = []
        url_queue = deque([(start_url, 0)])  # (url, depth)
        session_id = f"deepcrawl_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Statistics
        stats = {
            "total_urls_discovered": 0,
            "job_related_urls_found": 0,
            "pages_successfully_crawled": 0,
            "pages_failed": 0,
            "pages_skipped": 0,
            "max_depth_reached": 0
        }
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while url_queue and len(crawled_pages) < max_pages:
                current_url, depth = url_queue.popleft()
                
                # Skip if already visited or depth exceeded
                if current_url in visited_urls or depth > max_depth:
                    continue
                
                # Skip if robots.txt disallows or URL should be skipped
                if not is_allowed_by_robots(current_url) or should_skip_url(current_url):
                    reason = "Disallowed by robots.txt" if not is_allowed_by_robots(current_url) else "Skipped by pattern"
                    print(f"⚠️ Skipping {current_url} ({reason})", file=sys.stderr)
                    crawled_pages.append({
                        "url": current_url,
                        "depth": depth,
                        "status": "skipped",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })
                    stats["pages_skipped"] += 1
                    continue
                
                visited_urls.add(current_url)
                stats["max_depth_reached"] = max(stats["max_depth_reached"], depth)
                
                # Update progress
                progress = (len(crawled_pages) / max_pages) * 100
                send_progress_notification(
                    f"Crawling depth {depth}: {len(crawled_pages)}/{max_pages} pages", 
                    progress
                )
                
                print(f"🕷️ Crawling [{depth}] {current_url}", file=sys.stderr)
                
                try:
                    # Crawl the current URL
                    result = await crawler.arun(
                        url=current_url,
                        config=CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS,
                            session_id=session_id,
                        )
                    )
                    
                    if result.success and hasattr(result, 'html') and result.html:
                        # Quick check if this page contains job-related content
                        is_job_page = await quick_job_content_check(result.html, current_url)
                        
                        page_data = {
                            "url": current_url,
                            "depth": depth,
                            "html": result.html,
                            "html_length": len(result.html),
                            "is_job_related": is_job_page,
                            "timestamp": datetime.now().isoformat(),
                            "status": "success"
                        }
                        
                        crawled_pages.append(page_data)
                        stats["pages_successfully_crawled"] += 1
                        
                        if is_job_page:
                            stats["job_related_urls_found"] += 1
                        
                        # Extract and queue internal links for next depth level
                        if depth < max_depth:
                            # General internal links
                            internal_links = extract_links_from_html(result.html, current_url)
                            # Targeted job posting/listing links
                            targeted_job_links = extract_job_links_from_html(result.html, current_url)
                            stats["total_urls_discovered"] += len(internal_links)
                            
                            # Filter and prioritize job-related links
                            job_related_links = []
                            other_links = []
                            
                            # First, add targeted job links with highest priority
                            for link in targeted_job_links:
                                if link not in visited_urls:
                                    job_related_links.append(link)

                            # Then, classify remaining internal links
                            for link in internal_links:
                                if link not in visited_urls and link not in job_related_links:
                                    if is_job_related_url(link):
                                        job_related_links.append(link)
                                    else:
                                        other_links.append(link)

                            # If current page is a listing hub, further prioritize suspected detail URLs
                            listing_hint = any(x in current_url.lower() for x in [
                                '/offres-emploi', '/jobs', '/careers', '/emploi', '/recrutement'
                            ])
                            if listing_hint and job_related_links:
                                prioritized_details = [l for l in job_related_links if is_job_detail_url(l)]
                                non_details = [l for l in job_related_links if l not in prioritized_details]
                                job_related_links = prioritized_details + non_details
                            
                            # Add job-related links first, then others (slightly increased caps)
                            job_cap = 18 if depth == 0 else 12
                            other_cap = 8 if depth == 0 else 4
                            for link in job_related_links[:job_cap]:
                                url_queue.append((link, depth + 1))
                            for link in other_links[:other_cap]:
                                url_queue.append((link, depth + 1))

                            print(f"📋 Found {len(internal_links)} links, queued {min(len(job_related_links), job_cap) + min(len(other_links), other_cap)} (job={min(len(job_related_links), job_cap)}, other={min(len(other_links), other_cap)})", file=sys.stderr)
                        
                        print(f"✅ Successfully crawled: {current_url} (Job-related: {is_job_page})", file=sys.stderr)
                    
                    else:
                        error_msg = getattr(result, "error_message", "No HTML content returned")
                        print(f"❌ Failed to crawl: {current_url} - {error_msg}", file=sys.stderr)
                        
                        crawled_pages.append({
                            "url": current_url,
                            "depth": depth,
                            "error": error_msg,
                            "status": "failed",
                            "timestamp": datetime.now().isoformat()
                        })
                        stats["pages_failed"] += 1
                
                except Exception as e:
                    print(f"❌ Exception crawling {current_url}: {str(e)}", file=sys.stderr)
                    crawled_pages.append({
                        "url": current_url,
                        "depth": depth,
                        "error": f"Exception: {str(e)}",
                        "status": "failed",
                        "timestamp": datetime.now().isoformat()
                    })
                    stats["pages_failed"] += 1
                
                # Delay between requests to be respectful
                if delay_between_requests > 0 and len(crawled_pages) < max_pages:
                    await asyncio.sleep(delay_between_requests)
        
        send_progress_notification("Deep crawl completed", 100)
        
        print(f"🏁 Deep crawl completed:", file=sys.stderr)
        print(f"   - Total pages crawled: {len(crawled_pages)}", file=sys.stderr)
        print(f"   - Job-related pages: {stats['job_related_urls_found']}", file=sys.stderr)
        print(f"   - Max depth reached: {stats['max_depth_reached']}", file=sys.stderr)
        print(f"   - Total URLs discovered: {stats['total_urls_discovered']}", file=sys.stderr)
        
        return {
            "success": True,
            "start_url": start_url,
            "pages_crawled": len(crawled_pages),
            "pages": crawled_pages,
            "statistics": stats,
            "crawl_completed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"❌ Deep crawl error: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "start_url": start_url,
            "pages_crawled": len(crawled_pages) if 'crawled_pages' in locals() else 0,
            "pages": crawled_pages if 'crawled_pages' in locals() else []
        }

async def _fetch_url_html_with_pagination_impl(url: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Internal helper that performs the actual crawling and returns raw HTML pages.
    """
    session_id = f"fetchurl_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    results = []
    
    send_progress_notification(f"Début de l'extraction HTML...", 0)
    print(f"🌐 Starting crawler for URL: {url}", file=sys.stderr)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for page in range(1, max_pages + 1):
            send_progress_notification(f"Extraction page {page}/{max_pages}...", (page-1)/max_pages * 100)

            if max_pages > 1:
                crawl_url = f"{url}&page={page}" if '?' in url else f"{url}?page={page}"
            else:
                crawl_url = url

            print(f"🔍 Crawling URL: {crawl_url}", file=sys.stderr)

            try:
                result = await crawler.arun(
                    url=crawl_url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        session_id=session_id,
                    )
                )

                print(f"📊 Crawler result - Success: {result.success}", file=sys.stderr)
                if hasattr(result, 'html') and result.html:
                    print(f"📄 HTML length: {len(result.html)}", file=sys.stderr)

                if result.success and hasattr(result, 'html') and result.html:
                    results.append({
                        "page": page,
                        "url": crawl_url,
                        "html": result.html,
                        "status": "success"
                    })
                    print(f"✅ Page {page} crawled successfully", file=sys.stderr)
                else:
                    error_msg = getattr(result, "error_message", "No HTML content returned")
                    print(f"❌ Page {page} failed: {error_msg}", file=sys.stderr)
                    results.append({
                        "page": page,
                        "url": crawl_url,
                        "error": error_msg,
                        "status": "error"
                    })
            except Exception as crawl_error:
                print(f"❌ Crawler exception for page {page}: {crawl_error}", file=sys.stderr)
                results.append({
                    "page": page,
                    "url": crawl_url,
                    "error": f"Crawler exception: {str(crawl_error)}",
                    "status": "error"
                })

    send_progress_notification(f"Extraction terminée. {len(results)} pages traitées.", 100)

    # Ensure we always return at least 1 page result, even if empty
    if len(results) == 0:
        print(f"⚠️ Crawler returned 0 results, adding default empty page", file=sys.stderr)
        results.append({
            "page": 1,
            "url": url,
            "html": "",
            "status": "error",
            "error": "Crawler failed to retrieve content"
        })

    return {
        "success": True,
        "pages_fetched": len(results),
        "pages": results
    }


@mcp.tool(annotations={"title": "fetch_url_html_with_pagination"})
async def fetch_url_html_with_pagination(url: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Fait du crawling sur une URL (optionnellement paginée), retourne le HTML brut de chaque page.
    """
    try:
        return await _fetch_url_html_with_pagination_impl(url=url, max_pages=max_pages)
    except Exception as e:
        print(f"Error in fetch_url_html_with_pagination: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "pages_fetched": 0,
            "pages": []
        }


def _html_to_markdown_impl(html_content: str) -> Dict[str, Any]:
    """
    Internal helper to convert HTML content to Markdown format (non-MCP).
    """
    try:
        print(f"📝 HTML to Markdown input length: {len(html_content)}", file=sys.stderr)
        
        if not html_content or not html_content.strip():
            print("⚠️ Empty HTML content received", file=sys.stderr)
            return {
                "success": False,
                "error": "Empty HTML content provided",
                "markdown": "",
                "original_length": 0,
                "markdown_length": 0
            }
        
        original_length = len(html_content)
        print(f"📝 Starting conversion of {original_length} characters", file=sys.stderr)
        
        # Remove script and style tags
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert headings
        for i in range(1, 7):
            html_content = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'\n{"#" * i} \1\n', html_content, flags=re.IGNORECASE)
        
        # Convert paragraphs
        html_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert line breaks
        html_content = re.sub(r'<br[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        
        # Convert lists
        html_content = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all other HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\n\s*\n', '\n\n', html_content)
        markdown_content = html_content.strip()
        
        print(f"📝 Conversion complete: {len(markdown_content)} characters", file=sys.stderr)
        if len(markdown_content) > 0 and len(markdown_content) < 200:
            print(f"📝 Short markdown preview: {markdown_content}", file=sys.stderr)
        elif len(markdown_content) >= 200:
            print(f"📝 Markdown preview: {markdown_content[:200]}...", file=sys.stderr)
        
        return {
            "success": True,
            "markdown": markdown_content,
            "original_length": original_length,
            "markdown_length": len(markdown_content)
        }
    except Exception as e:
        print(f"❌ HTML to Markdown conversion error: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": f"Failed to convert HTML to Markdown: {str(e)}"
        }


@mcp.tool(annotations={"title": "html_to_markdown"})
async def html_to_markdown(html_content: str) -> Dict[str, Any]:
    """
    Convert HTML content to Markdown format.
    """
    # Delegate to non-MCP helper to keep callable semantics inside this module
    return _html_to_markdown_impl(html_content)


@mcp.tool(annotations={"title": "extract_jobs"})
async def extract_jobs(url: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Combined tool: fetch HTML for the given URL (with optional pagination) and convert each page to Markdown.

    Returns a structured result including per-page HTML/Markdown and a concatenated markdown string.
    """
    try:
        # Step 1: Fetch pages via helper (avoid calling MCP-decorated function directly)
        fetch_result = await _fetch_url_html_with_pagination_impl(url=url, max_pages=max_pages)
        pages = fetch_result.get("pages", []) if isinstance(fetch_result, dict) else []

        output_pages: List[Dict[str, Any]] = []
        combined_md_chunks: List[str] = []

        # Step 2: Convert each successful page's HTML to markdown
        for p in pages:
            page_url = p.get("url")
            page_num = p.get("page")
            status = p.get("status")
            html = p.get("html") or ""

            page_out: Dict[str, Any] = {
                "page": page_num,
                "url": page_url,
                "status": status,
                "html_length": len(html) if html else 0,
            }

            if status == "success" and html:
                md_result = _html_to_markdown_impl(html)
                page_out["markdown_success"] = md_result.get("success", False)
                page_out["markdown_length"] = md_result.get("markdown_length", 0)
                if md_result.get("success"):
                    md = md_result.get("markdown", "")
                    page_out["markdown_preview"] = (md[:200] + "...") if len(md) > 200 else md
                    combined_md_chunks.append(md)
                else:
                    page_out["markdown_error"] = md_result.get("error", "Unknown markdown error")
            else:
                page_out["markdown_success"] = False
                page_out["markdown_length"] = 0
                if p.get("error"):
                    page_out["fetch_error"] = p.get("error")

            output_pages.append(page_out)

        combined_markdown = "\n\n".join(chunk for chunk in combined_md_chunks if chunk)
        success = len(combined_markdown.strip()) > 0

        return {
            "success": success,
            "url": url,
            "pages_fetched": len(pages),
            "pages_with_markdown": sum(1 for pg in output_pages if pg.get("markdown_success")),
            "combined_markdown_length": len(combined_markdown),
            "combined_markdown": combined_markdown,
            "pages": output_pages,
        }
    except Exception as e:
        print(f"❌ extract_jobs error: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "pages": []
        }


if __name__ == "__main__":
    import traceback
    print("Starting JobOfferServer with FastMCP...", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server stopped due to: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)