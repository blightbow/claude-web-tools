"""Kagi Search MCP Server - Exposes Kagi Search and Summarizer APIs for Claude Code."""

import argparse
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

import httpx
from bs4 import BeautifulSoup
from kagiapi import KagiClient
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("kagi")

CONFIG_PATH = Path.home() / ".config" / "kagi" / "api_key"

# Profile-specific tool descriptions for different Claude clients
TOOL_DESCRIPTIONS = {
    "search": {
        "code": """Search the web using Kagi's curated search index.

Use this as an alternative to the built-in WebSearch tool when WebSearch
returns few or poor quality results. Kagi's index is independently curated,
resistant to SEO spam, and may surface different sources. Returns raw search
results with snippets and timestamps, plus related search suggestions.""",
        "desktop": """Search the web using Kagi's curated search index.

Use this as an alternative to the built-in web_search tool when web_search
returns few or poor quality results. Kagi's index is independently curated,
resistant to SEO spam, and may surface different sources. Returns raw search
results with snippets and timestamps, plus related search suggestions.""",
    },
    "summarize": {
        "code": """Summarize content from a URL or text using Kagi's Universal Summarizer.

Supports web pages, PDFs, YouTube videos, audio files, and documents.
Use this when WebFetch fails due to agent blacklisting or access restrictions.""",
        "desktop": """Summarize content from a URL or text using Kagi's Universal Summarizer.

Supports web pages, PDFs, YouTube videos, audio files, and documents.
Use this when web_fetch fails due to agent blacklisting or access restrictions.""",
    },
    # web_fetch_direct is desktop-only (registered conditionally in main)
}


def apply_profile(profile: str) -> None:
    """Apply tool descriptions for the specified profile.

    Note: Uses _tool_manager, a private FastMCP API.
    """
    for tool_name, descriptions in TOOL_DESCRIPTIONS.items():
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool:
            tool.description = descriptions[profile]


def get_api_key() -> str:
    """Load API key from config file or environment."""
    # Environment variable takes precedence
    if key := os.environ.get("KAGI_API_KEY"):
        return key
    # Fall back to config file
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text().strip()
    return ""


def get_client() -> Optional[KagiClient]:
    """Create a Kagi client with the configured API key."""
    api_key = get_api_key()
    if not api_key:
        return None
    return KagiClient(api_key=api_key)


@mcp.tool()
async def search(query: str, limit: int = 5) -> str:
    """Search the web using Kagi's curated search index.

    Use this as an alternative to the built-in WebSearch tool when WebSearch
    returns few or poor quality results. Kagi's index is independently curated,
    resistant to SEO spam, and may surface different sources. Returns raw search
    results with snippets and timestamps, plus related search suggestions.

    Args:
        query: The search query
        limit: Maximum number of results to return (default 5)
    """
    client = get_client()
    if not client:
        return "Error: API key not found. Create ~/.config/kagi/api_key or set KAGI_API_KEY env var."

    try:
        response = client.search(query, limit=limit)
    except Exception as e:
        logger.exception("Error during search")
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "Error: Invalid API key. Check ~/.config/kagi/api_key or KAGI_API_KEY env var."
        if "402" in error_msg:
            return "Error: Insufficient API credits. Add funds at https://kagi.com/settings/billing"
        return f"Error: {error_msg}"

    # Parse results
    results = []
    related_searches = []

    for item in response.get("data", []):
        item_type = item.get("t")

        if item_type == 0:  # Search result
            title = item.get("title", "Untitled")
            item_url = item.get("url", "")
            snippet = item.get("snippet", "")
            published = item.get("published")

            # Format as markdown
            if published:
                results.append(f"[{title}]({item_url}) - {snippet} ({published})")
            else:
                results.append(f"[{title}]({item_url}) - {snippet}")

        elif item_type == 1:  # Related searches
            related_searches = item.get("list", [])

    # Build output
    output_parts = []

    if results:
        output_parts.append("Results:")
        for i, result in enumerate(results, 1):
            output_parts.append(f"{i}. {result}")
    else:
        output_parts.append("No results found.")

    if related_searches:
        output_parts.append("")
        output_parts.append(f"Related searches: {', '.join(related_searches)}")

    return "\n".join(output_parts)


@mcp.tool()
async def summarize(
    url: Optional[str] = None,
    text: Optional[str] = None,
    summary_type: str = "summary"
) -> str:
    """Summarize content from a URL or text using Kagi's Universal Summarizer.

    Supports web pages, PDFs, YouTube videos, audio files, and documents.
    Use this when WebFetch fails due to agent blacklisting or access restrictions.

    Args:
        url: URL to summarize (PDFs, YouTube, articles, audio)
        text: Raw text to summarize (alternative to url)
        summary_type: Output format - "summary" for prose, "takeaway" for bullet points
    """
    client = get_client()
    if not client:
        return "Error: API key not found. Create ~/.config/kagi/api_key or set KAGI_API_KEY env var."

    if not url and not text:
        return "Error: Either 'url' or 'text' must be provided."

    if url and text:
        return "Error: Provide either 'url' or 'text', not both."

    if summary_type not in ("summary", "takeaway"):
        return "Error: summary_type must be 'summary' or 'takeaway'."

    try:
        if url:
            response = client.summarize(url=url, summary_type=summary_type, target_language="EN")
        else:
            response = client.summarize(text=text, summary_type=summary_type, target_language="EN")
    except Exception as e:
        logger.exception("Error during summarization")
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "Error: Invalid API key. Check ~/.config/kagi/api_key or KAGI_API_KEY env var."
        if "402" in error_msg:
            return "Error: Insufficient API credits. Add funds at https://kagi.com/settings/billing"
        return f"Error: {error_msg}"

    # Extract summary
    output = response.get("data", {}).get("output", "")

    if not output:
        return "Error: No summary returned from API."

    return output


# Default headers for web_fetch_direct
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _extract_text_spans(soup: BeautifulSoup) -> list[str]:
    """Extract text content from HTML, split into spans by block elements."""
    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Find main content area if possible
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup

    spans = []
    # Block elements that should create span boundaries
    block_tags = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
                  "li", "td", "th", "blockquote", "pre", "figcaption"}

    for element in main.find_all(block_tags):
        text = element.get_text(separator=" ", strip=True)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if text and len(text) > 20:  # Skip very short fragments
            spans.append(text)

    # Deduplicate while preserving order (nested elements can cause duplicates)
    seen = set()
    unique_spans = []
    for span in spans:
        if span not in seen:
            seen.add(span)
            unique_spans.append(span)

    return unique_spans


def _build_document_xml(
    title: str,
    spans: list[str],
    url: str,
    mime_type: str,
    doc_index: int = 1
) -> str:
    """Build XML document structure matching Claude Desktop's format."""
    lines = [f'<document index="{doc_index}">']
    lines.append(f"  <source>{xml_escape(title)}</source>")
    lines.append("  <document_content>")

    for i, span_text in enumerate(spans, 1):
        span_index = f"{doc_index}-{i}"
        lines.append(f'    <span index="{span_index}">{xml_escape(span_text)}</span>')

    lines.append("  </document_content>")
    lines.append('  <metadata key="content_type">html</metadata>')
    lines.append(f'  <metadata key="destination_url">{xml_escape(url)}</metadata>')
    lines.append(f'  <metadata key="mime_type">{xml_escape(mime_type)}</metadata>')
    lines.append("</document>")

    return "\n".join(lines)


# Desktop-only tool - registered conditionally in main()
async def web_fetch_direct(url: str, max_tokens: Optional[int] = None) -> str:
    """Fetch raw HTML content from a URL without summarization.

    Use this as an alternative to web_fetch when web_fetch fails due to
    agent blacklisting or access restrictions. Returns full page content
    in XML document format with span indices for citation.

    Args:
        url: The URL to fetch
        max_tokens: Optional limit on content length (approximate token count)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url, headers=_FETCH_HEADERS)
            response.raise_for_status()
    except httpx.TimeoutException:
        return f"Error: Request timed out for {url}"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: Failed to fetch {url} - {type(e).__name__}"

    # Check content type
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return f"Error: Unsupported content type '{content_type}'. Only HTML is supported."

    # Parse HTML
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        return f"Error: Failed to parse HTML - {e}"

    # Extract title (prefer <title>, fallback to <h1>)
    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Extract spans
    spans = _extract_text_spans(soup)

    if not spans:
        return f"Error: No content extracted from {url}"

    # Apply token limit if specified (approximate: ~4 chars per token)
    if max_tokens:
        char_limit = max_tokens * 4
        total_chars = 0
        truncated_spans = []
        for span in spans:
            if total_chars + len(span) > char_limit:
                truncated_spans.append("[content truncated]")
                break
            truncated_spans.append(span)
            total_chars += len(span)
        spans = truncated_spans

    # Build XML output
    mime_type = content_type.split(";")[0].strip() if content_type else "text/html"
    return _build_document_xml(title, spans, str(response.url), mime_type)


# NOTE: search_and_summarize is commented out to reduce API costs.
# It performs 1 search + N summarize calls, which adds up quickly.
# Uncomment to re-enable.
#
# @mcp.tool()
# async def search_and_summarize(
#     query: str,
#     limit: int = 3,
#     summary_type: str = "takeaway"
# ) -> str:
#     """Search the web and summarize top results for a synthesized overview.
#
#     Combines Kagi Search with the Universal Summarizer to provide deeper
#     insight than search snippets alone. Each result URL is summarized to
#     extract key points.
#
#     Args:
#         query: The search query
#         limit: Number of results to summarize (default 3, max 5)
#         summary_type: "takeaway" for bullet points (default), "summary" for prose
#     """
#     client = get_client()
#     if not client:
#         return "Error: API key not found. Create ~/.config/kagi/api_key or set KAGI_API_KEY env var."
#
#     # Cap limit to avoid excessive API usage
#     limit = min(limit, 5)
#
#     if summary_type not in ("summary", "takeaway"):
#         return "Error: summary_type must be 'summary' or 'takeaway'."
#
#     # Step 1: Perform search
#     try:
#         search_response = client.search(query, limit=limit)
#     except Exception as e:
#         logger.exception("Error during search")
#         error_msg = str(e)
#         if "401" in error_msg or "Unauthorized" in error_msg:
#             return "Error: Invalid API key. Check ~/.config/kagi/api_key or KAGI_API_KEY env var."
#         if "402" in error_msg:
#             return "Error: Insufficient API credits. Add funds at https://kagi.com/settings/billing"
#         return f"Error during search: {error_msg}"
#
#     # Parse search results
#     results = []
#     related_searches = []
#
#     for item in search_response.get("data", []):
#         item_type = item.get("t")
#
#         if item_type == 0:  # Search result
#             results.append({
#                 "title": item.get("title", "Untitled"),
#                 "url": item.get("url", ""),
#                 "snippet": item.get("snippet", ""),
#                 "published": item.get("published"),
#             })
#         elif item_type == 1:  # Related searches
#             related_searches = item.get("list", [])
#
#     if not results:
#         return "No results found."
#
#     # Step 2: Summarize each result URL in parallel
#     async def summarize_url(result: dict) -> dict:
#         """Summarize a single URL, returning result with summary."""
#         url = result["url"]
#         if not url:
#             return {**result, "summary": None, "error": "No URL"}
#
#         try:
#             response = await asyncio.to_thread(
#                 client.summarize,
#                 url=url,
#                 summary_type=summary_type,
#                 target_language="EN"
#             )
#             summary = response.get("data", {}).get("output", "")
#             return {**result, "summary": summary, "error": None}
#         except Exception as e:
#             logger.warning(f"Failed to summarize {url}: {e}")
#             return {**result, "summary": None, "error": str(e)}
#
#     summarized_results = await asyncio.gather(*[summarize_url(r) for r in results])
#
#     # Step 3: Build formatted output
#     output_parts = []
#     output_parts.append(f"# Search: {query}\n")
#
#     # Sources section
#     output_parts.append("## Sources\n")
#     for i, result in enumerate(summarized_results, 1):
#         title = result["title"]
#         url = result["url"]
#         published = result.get("published")
#         if published:
#             output_parts.append(f"{i}. [{title}]({url}) ({published})")
#         else:
#             output_parts.append(f"{i}. [{title}]({url})")
#     output_parts.append("")
#
#     # Key findings section
#     output_parts.append("## Key Findings\n")
#     for i, result in enumerate(summarized_results, 1):
#         title = result["title"]
#         summary = result.get("summary")
#         error = result.get("error")
#
#         output_parts.append(f"### {i}. {title}\n")
#
#         if summary:
#             output_parts.append(summary)
#         elif error:
#             output_parts.append(f"*Could not summarize: {error}*")
#         else:
#             # Fall back to snippet
#             snippet = result.get("snippet", "No content available.")
#             output_parts.append(f"*{snippet}*")
#
#         output_parts.append("")
#
#     # Related searches
#     if related_searches:
#         output_parts.append("## Related Searches\n")
#         output_parts.append(", ".join(related_searches))
#
#     return "\n".join(output_parts)


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="Kagi MCP Server")
    parser.add_argument(
        "--profile",
        choices=["code", "desktop"],
        default="code",
        help="Target client profile (default: code)",
    )
    args = parser.parse_args()

    apply_profile(args.profile)

    # Register desktop-only tools
    if args.profile == "desktop":
        mcp.add_tool(web_fetch_direct)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
