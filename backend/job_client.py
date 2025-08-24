import ast
import asyncio
import json
import pprint
import sys
import time
from typing import Dict, Any, List, Optional
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from config import settings
import logging
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=2, width=100)

def unwrap_tool_result(resp):
    """Unwrap tool result from MCP response"""
    if hasattr(resp, "structured_content") and resp.structured_content:
        return resp.structured_content.get("result", resp.structured_content)
    if hasattr(resp, "content") and resp.content:
        text = resp.content
        try:
            return ast.literal_eval(text)
        except Exception:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return resp


class JobCrawlerClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.client = None
        self.tools = []
        self.is_connected = False
        # Prevent duplicate discovery/logs across multiple calls
        self._tools_discovered = False

    async def start(self):
        """Initialize MCP client connection"""
        try:
            transport = StdioTransport(
                command=sys.executable,
                args=[self.server_script],
            )
            self.client = Client(transport)
            print(f"🚀 Connecting to Job Crawler MCP server: {self.server_script}")
            await self.client.__aenter__()

            # Discover available tools
            await self.discover_tools()
            self.is_connected = True
            tool_names = [t.name for t in self.tools]
            print(f"✅ Connected! Available tools: {tool_names}")

        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            raise

    async def stop(self):
        """Close MCP client connection"""
        if self.client and self.is_connected:
            await self.client.__aexit__(None, None, None)
            self.is_connected = False
            print("🔌 Disconnected from MCP server")
            # Reset discovery flag on disconnect
            self._tools_discovered = False

    async def discover_tools(self):
        """Discover available tools from the MCP server"""
        if not self.client:
            raise RuntimeError("Client not connected. Call start() first.")

        # Idempotent: if already discovered and cache exists, skip re-discovery/log
        if self._tools_discovered and self.tools:
            return self.tools

        print("🛠️  Discovering tools...")
        self.tools = await asyncio.wait_for(
            self.client.list_tools(), timeout=settings.MCP_TOOL_TIMEOUT
        )
        self._tools_discovered = True
        return self.tools

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client or not self.is_connected:
            raise RuntimeError("Client not connected. Call start() first.")

        try:
            logger.info(f"Calling tool: {tool_name} with params: {params}")
            start_time = time.time()
            result = await asyncio.wait_for(
                self.client.call_tool(tool_name, params),
                timeout=settings.MCP_TOOL_TIMEOUT,
            )
            unwrapped = unwrap_tool_result(result)

            elapsed = time.time() - start_time
            logger.info(f"Tool {tool_name} completed in {elapsed:.2f}s")
            return {"success": True, "result": unwrapped, "elapsed_time": elapsed}

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"Tool {tool_name} timed out after {elapsed:.2f}s")
            return {"success": False, "error": "timeout", "elapsed_time": elapsed}
        except Exception as e:
            if "notifications/progress" in str(e).lower():
                logger.warning(f"Ignoring progress notification: {str(e)}")
                try:
                    result = await asyncio.wait_for(
                        self.client.call_tool(tool_name, params),
                        timeout=settings.MCP_TOOL_TIMEOUT,
                    )  # Retry once
                    unwrapped = unwrap_tool_result(result)
                    elapsed = time.time() - start_time
                    logger.info(f"Tool {tool_name} completed after retry in {elapsed:.2f}s")
                    return {"success": True, "result": unwrapped, "elapsed_time": elapsed}
                except Exception as retry_e:
                    logger.error(f"Tool {tool_name} failed after retry: {str(retry_e)}")
                    return {"success": False, "error": str(retry_e)}
            else:
                logger.error(f"Tool {tool_name} failed: {str(e)}")
                return {"success": False, "error": str(e)}

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return getattr(tool, "inputSchema", {})
        return None

    def list_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]

    async def interactive_prompt(self):
        """Interactive mode for testing job crawler tools"""
        print("\n🕷️  === Job Crawler Client Interactive Mode ===")
        print("Quick Commands:")
        print("  company <name>     - Find company website and job URLs")
        print("  check <url>        - Quick check if URL has job content")
        print("  crawl <url>        - Deep crawl job pages from URL")
        print("  fetch <url>        - Fetch HTML from URL")
        print("  markdown <html>    - Convert HTML to markdown")
        print("\nGeneric Commands:")
        print("  list              - List all available tools")
        print("  call <number>     - Call tool by number")
        print("  schema <number>   - Show tool schema")
        print("  exit              - Exit client")

        while True:
            try:
                cmd = input("\n🤖 job-crawler> ").strip()

                if cmd == "exit":
                    print("👋 Exiting client...")
                    break

                elif cmd == "list":
                    await self._list_tools()

                elif cmd.startswith("company "):
                    company_name = cmd[8:].strip()
                    if company_name:
                        await self._quick_company_search(company_name)
                    else:
                        print("❌ Please provide a company name")

                elif cmd.startswith("check "):
                    url = cmd[6:].strip()
                    if url:
                        await self._quick_url_check(url)
                    else:
                        print("❌ Please provide a URL")

                elif cmd.startswith("crawl "):
                    url = cmd[6:].strip()
                    if url:
                        await self._quick_deep_crawl(url)
                    else:
                        print("❌ Please provide a URL")

                elif cmd.startswith("fetch "):
                    url = cmd[6:].strip()
                    if url:
                        await self._quick_fetch(url)
                    else:
                        print("❌ Please provide a URL")

                elif cmd.startswith("markdown "):
                    html_content = cmd[9:].strip()
                    if html_content:
                        await self._quick_markdown(html_content)
                    else:
                        print("❌ Please provide HTML content")

                # 'analyze' command removed

                elif cmd.startswith("schema "):
                    await self._show_schema(cmd)

                elif cmd.startswith("call "):
                    await self._handle_tool_call(cmd)

                else:
                    print("❌ Unknown command. Type one of the commands above.")

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def _list_tools(self):
        """List all available tools with descriptions"""
        print("\n📋 Available Tools:")
        for i, tool in enumerate(self.tools):
            desc = getattr(tool, "description", "") or getattr(tool, "title", "")
            print(f"  {i+1:2}. {tool.name:<35} - {desc}")

    async def _quick_company_search(self, company_name: str):
        """Quick company website and job URL search"""
        print(f"🔍 Searching for {company_name}...")
        result = await self.call_tool("get_official_website_and_generate_job_urls", {"company_name": company_name})

        if result["success"]:
            data = result["result"]
            if data.get("success"):
                print(f"✅ Found official website: {data['official_website']}")
                print(f"📋 Generated {len(data['possible_job_urls'])} potential job URLs:")
                for i, url in enumerate(data['possible_job_urls'], 1):
                    print(f"   {i}. {url}")
            else:
                print(f"❌ Search failed: {data.get('reason', 'Unknown error')}")
        else:
            print(f"❌ Tool call failed: {result['error']}")

    async def _quick_url_check(self, url: str):
        """Quick URL job content check"""
        print(f"🔍 Checking {url}...")
        result = await self.call_tool("precheck_job_offer_url", {"url": url})

        if result["success"]:
            data = result["result"]
            if data.get("success"):
                is_job = data.get("is_offer_page", False)
                status = "✅ JOB PAGE" if is_job else "❌ NOT A JOB PAGE"
                print(f"{status}")
                print(f"   Patterns found: {data.get('patterns_found', 0)}")
                if data.get('matched_patterns'):
                    print(f"   Matched patterns: {', '.join(data['matched_patterns'])}")
                if data.get('snippet'):
                    print(f"   Content preview: {data['snippet'][:100]}...")
            else:
                print(f"❌ Check failed: {data.get('reason', 'Unknown error')}")
        else:
            print(f"❌ Tool call failed: {result['error']}")

    async def _quick_deep_crawl(self, url: str):
        """Quick deep crawl with default settings"""
        print(f"🕷️  Starting deep crawl from {url}...")
        print("⚙️  Using default settings (max_pages=10, max_depth=2, delay=1.0s)")

        result = await self.call_tool("deep_crawl_job_pages", {
            "start_url": url,
            "max_pages": 10,
            "max_depth": 2,
            "delay_between_requests": 1.0
        })

        if result["success"]:
            data = result["result"]
            if data.get("success"):
                stats = data.get("statistics", {})
                print(f"\n📊 Crawl Results:")
                print(f"   ⏱️  Time taken: {result.get('elapsed_time', 0):.2f}s")
                print(f"   📄 Pages crawled: {data.get('pages_crawled', 0)}")
                print(f"   ✅ Job-related pages: {stats.get('job_related_urls_found', 0)}")
                print(f"   🔗 Total URLs discovered: {stats.get('total_urls_discovered', 0)}")
                print(f"   📊 Max depth reached: {stats.get('max_depth_reached', 0)}")
                print(f"   ❌ Failed crawls: {stats.get('pages_failed', 0)}")

                # Show some job-related URLs found
                job_pages = [p for p in data.get("pages", []) if p.get("is_job_related")]
                if job_pages:
                    print(f"\n🎯 Job-related pages found:")
                    for i, page in enumerate(job_pages[:5], 1):
                        print(f"   {i}. [{page.get('depth', 0)}] {page.get('url', '')}")

                # Offer to analyze results
                print(f"\n💡 Tip: You can analyze these results with 'analyze' command")
            else:
                print(f"❌ Crawl failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Tool call failed: {result['error']}")

    async def _quick_fetch(self, url: str):
        """Quick HTML fetch"""
        print(f"📄 Fetching HTML from {url}...")
        result = await self.call_tool("fetch_url_html_with_pagination", {"url": url, "max_pages": 1})

        if result["success"]:
            data = result["result"]
            if data.get("success"):
                pages = data.get("pages", [])
                if pages:
                    page = pages[0]
                    if page.get("status") == "success":
                        html_length = len(page.get("html", ""))
                        print(f"✅ HTML fetched successfully")
                        print(f"   📊 Content length: {html_length:,} characters")
                        print(f"   🔗 URL: {page.get('url', '')}")

                        if html_length < 500:
                            print(f"   📄 Content preview:")
                            print(f"   {page.get('html', '')[:200]}...")
                    else:
                        print(f"❌ Fetch failed: {page.get('error', 'Unknown error')}")
                else:
                    print("❌ No pages returned")
            else:
                print(f"❌ Fetch failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Tool call failed: {result['error']}")

    async def _quick_markdown(self, html_content: str):
        """Quick HTML to markdown conversion"""
        print(f"📝 Converting HTML to markdown...")
        result = await self.call_tool("html_to_markdown", {"html_content": html_content})

        if result["success"]:
            data = result["result"]
            if data.get("success"):
                markdown = data.get("markdown", "")
                print(f"✅ Conversion successful")
                print(f"   📊 Original length: {data.get('original_length', 0):,} chars")
                print(f"   📊 Markdown length: {data.get('markdown_length', 0):,} chars")
                print(f"   📄 Markdown preview:")
                print(f"   {markdown[:300]}...")
            else:
                print(f"❌ Conversion failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Tool call failed: {result['error']}")

    # _quick_analyze removed (tool analyze_crawled_pages deprecated)

    async def _show_schema(self, cmd: str):
        """Show tool schema"""
        try:
            idx = int(cmd.split()[1]) - 1
            if idx < 0 or idx >= len(self.tools):
                print("❌ Invalid tool number")
                return
            tool = self.tools[idx]
            schema = getattr(tool, "inputSchema", {})
            print(f"\n📋 Schema for {tool.name}:")
            pp.pprint(schema)
        except Exception as e:
            print(f"❌ Error: {e}")

    async def _handle_tool_call(self, cmd: str):
        """Handle interactive tool calling"""
        try:
            idx = int(cmd.split()[1]) - 1
            if idx < 0 or idx >= len(self.tools):
                print("❌ Invalid tool number")
                return

            tool = self.tools[idx]
            schema = getattr(tool, "inputSchema", {})

            print(f"\n🔧 Calling tool: {tool.name}")

            if schema and "properties" in schema:
                params = {}
                for pname, pinfo in schema["properties"].items():
                    required = pname in schema.get("required", [])
                    ptype = pinfo.get("type", "string")
                    default = pinfo.get("default", None)
                    description = pinfo.get("description", "")

                    if not required and default is not None:
                        prompt = f"📝 {pname} ({ptype}, default={default}): {description}\n   > "
                    else:
                        prompt = f"📝 {pname} ({ptype}{'*' if required else ''}): {description}\n   > "

                    val = input(prompt).strip()

                    if val == "" and not required and default is not None:
                        val = default
                    elif val == "" and required:
                        print(f"❌ '{pname}' is required, try again.")
                        return

                    # Type conversion
                    if ptype == "integer" and val:
                        try:
                            val = int(val)
                        except ValueError:
                            print(f"❌ Invalid integer for '{pname}', try again.")
                            return
                    elif ptype == "number" and val:
                        try:
                            val = float(val)
                        except ValueError:
                            print(f"❌ Invalid number for '{pname}', try again.")
                            return
                    elif ptype == "boolean" and val:
                        val = str(val).lower() in ("true", "yes", "1", "on")
                    elif ptype == "array" and val:
                        try:
                            val = json.loads(val) if val.startswith('[') else val.split(',')
                        except json.JSONDecodeError:
                            val = val.split(',')

                    params[pname] = val

                print(f"\n🚀 Executing {tool.name}...")
                result = await self.call_tool(tool.name, params)

                print(f"\n📋 Result (took {result.get('elapsed_time', 0):.2f}s):")
                if result["success"]:
                    # Pretty print with better formatting for job crawler results
                    self._pretty_print_result(result["result"], tool.name)
                else:
                    print(f"❌ Error: {result['error']}")

            else:
                print("ℹ️  Tool has no input schema - calling with no parameters.")
                result = await self.call_tool(tool.name, {})
                print(f"\n📋 Result (took {result.get('elapsed_time', 0):.2f}s):")
                if result["success"]:
                    self._pretty_print_result(result["result"], tool.name)
                else:
                    print(f"❌ Error: {result['error']}")

        except Exception as e:
            print(f"❌ Error calling tool: {e}")

    def _pretty_print_result(self, result: Any, tool_name: str):
        """Pretty print results with special formatting for job crawler tools"""
        if tool_name == "deep_crawl_job_pages" and isinstance(result, dict):
            if result.get("success"):
                print("✅ Deep crawl completed successfully")
                stats = result.get("statistics", {})
                print(f"📊 Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")

                pages = result.get("pages", [])
                job_pages = [p for p in pages if p.get("is_job_related")]
                print(f"\n🎯 Job-related pages ({len(job_pages)}):")
                for page in job_pages[:3]:  # Show first 3
                    print(f"   - [{page.get('depth')}] {page.get('url')}")
                if len(job_pages) > 3:
                    print(f"   ... and {len(job_pages) - 3} more")
            else:
                print(f"❌ Deep crawl failed: {result.get('error')}")

        # pretty-print branch for analyze_crawled_pages removed
        else:
            # Default pretty print
            pp.pprint(result)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


class LegacyMCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.client = None
        self.tools = []
        self.is_connected = False

    async def start(self):
        """Initialize MCP client connection"""
        try:
            transport = StdioTransport(
                command=sys.executable,
                args=[self.server_script],
            )
            self.client = Client(transport)
            print(f"🚀 Connecting to MCP server: {self.server_script}")
            await self.client.__aenter__()
            
            # Discover available tools
            await self.discover_tools()
            self.is_connected = True
            tool_names = [t.name for t in self.tools]
            print(f"✅ Connected! Available tools: {tool_names}")
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            raise

    async def stop(self):
        """Close MCP client connection"""
        if self.client and self.is_connected:
            await self.client.__aexit__(None, None, None)
            self.is_connected = False
            print("🔌 Disconnected from MCP server")

    async def discover_tools(self):
        """Discover available tools from the MCP server"""
        if not self.client:
            raise RuntimeError("Client not connected. Call start() first.")
        
        print("🛠️  Discovering tools...")
        self.tools = await asyncio.wait_for(
            self.client.list_tools(), timeout=settings.MCP_TOOL_TIMEOUT
        )
        return self.tools

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client or not self.is_connected:
           raise RuntimeError("Client not connected. Call start() first.")
    
        try:
           logger.info(f"Calling tool: {tool_name} with params: {params}")
           result = await asyncio.wait_for(
               self.client.call_tool(tool_name, params),
               timeout=settings.MCP_TOOL_TIMEOUT,
           )
           unwrapped = unwrap_tool_result(result)
           logger.info(f"Tool {tool_name} completed successfully")
           return {"success": True, "result": unwrapped}
    
        except asyncio.TimeoutError:
           logger.error(f"Tool {tool_name} timed out")
           return {"success": False, "error": "timeout"}
        except Exception as e:
            if "notifications/progress" in str(e).lower():
               logger.warning(f"Ignoring progress notification: {str(e)}")
               try:
                  result = await asyncio.wait_for(
                      self.client.call_tool(tool_name, params),
                      timeout=settings.MCP_TOOL_TIMEOUT,
                  )  # Retry once
                  unwrapped = unwrap_tool_result(result)
                  logger.info(f"Tool {tool_name} completed successfully after retry")
                  return {"success": True, "result": unwrapped}
               except Exception as retry_e:
                  logger.error(f"Tool {tool_name} failed after retry: {str(retry_e)}")
                  return {"success": False, "error": str(retry_e)}
            else:
                logger.error(f"Tool {tool_name} failed: {str(e)}")
                return {"success": False, "error": str(e)}

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return getattr(tool, "inputSchema", {})
        return None

    def list_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]

    async def interactive_prompt(self):
        """Interactive mode for testing tools"""
        print("\n--- MCP Client Interactive ---")
        print("Commands: list, call <number>, schema <number>, exit")
        
        while True:
            try:
                cmd = input("Enter command> ").strip()
                
                if cmd == "exit":
                    print("Exiting client...")
                    await self.stop()
                    break
                    
                elif cmd == "list":
                    for i, t in enumerate(self.tools):
                        desc = getattr(t, "description", None) or getattr(t, "title", "")
                        print(f"{i+1}. {t.name} - {desc}")
                        
                elif cmd.startswith("schema "):
                    try:
                        idx = int(cmd.split()[1]) - 1
                        if idx < 0 or idx >= len(self.tools):
                            print("Invalid tool number")
                            continue
                        tool = self.tools[idx]
                        schema = getattr(tool, "inputSchema", {})
                        print(f"Schema for {tool.name}:")
                        pp.pprint(schema)
                    except Exception as e:
                        print(f"Error: {e}")
                        
                elif cmd.startswith("call "):
                    await self._handle_tool_call(cmd)
                    
                else:
                    print("Unknown command. Available: list, call <number>, schema <number>, exit")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def _handle_tool_call(self, cmd: str):
        """Handle interactive tool calling"""
        try:
            idx = int(cmd.split()[1]) - 1
            if idx < 0 or idx >= len(self.tools):
                print("Invalid tool number")
                return
                
            tool = self.tools[idx]
            schema = getattr(tool, "inputSchema", {})
            
            if schema and "properties" in schema:
                params = {}
                for pname, pinfo in schema["properties"].items():
                    required = pname in schema.get("required", [])
                    ptype = pinfo.get("type", "string")
                    default = pinfo.get("default", None)
                    description = pinfo.get("description", "")
                    
                    if not required and default is not None:
                        prompt = f"Enter '{pname}' ({ptype}, default={default}): {description}\n> "
                    else:
                        prompt = f"Enter '{pname}' ({ptype}{'*' if required else ''}): {description}\n> "
                        
                    val = input(prompt).strip()
                    
                    if val == "" and not required and default is not None:
                        val = default
                    elif val == "" and required:
                        print(f"'{pname}' is required, try again.")
                        return
                        
                    # Type conversion
                    if ptype == "integer" and val:
                        try:
                            val = int(val)
                        except ValueError:
                            print(f"Invalid integer for '{pname}', try again.")
                            return
                    elif ptype == "number" and val:
                        try:
                            val = float(val)
                        except ValueError:
                            print(f"Invalid number for '{pname}', try again.")
                            return
                    elif ptype == "boolean" and val:
                        val = str(val).lower() in ("true", "yes", "1", "on")
                    elif ptype == "array" and val:
                        try:
                            val = json.loads(val) if val.startswith('[') else val.split(',')
                        except json.JSONDecodeError:
                            val = val.split(',')
                            
                    params[pname] = val
                    
                print(f"\n🔧 Calling {tool.name} with params: {params}")
                result = await self.call_tool(tool.name, params)
                print("\n📋 Result:")
                pp.pprint(result)
                
            else:
                print("Tool has no input schema - calling with no parameters.")
                result = await self.call_tool(tool.name, {})
                print("\n📋 Result:")
                pp.pprint(result)
                
        except Exception as e:
            print(f"Error calling tool: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


async def main():
    """Main function for standalone testing"""
    if len(sys.argv) < 2:
        print("Usage: python job_client.py <mcp_server.py>")
        print("Example: python job_client.py server.py")
        sys.exit(1)
        
    server_script_path = sys.argv[1]
    
    # Use the merged JobCrawlerClient
    async with JobCrawlerClient(server_script_path) as client:
        await client.interactive_prompt()


if __name__ == "__main__":
    asyncio.run(main())

# Re-export MCPClient as a thin shim to the JobCrawlerClient for backward compatibility
class MCPClient(JobCrawlerClient):  # type: ignore[misc]
    """Compatibility shim: import MCPClient from job_client to get the JobCrawlerClient."""
    pass
