# Kagi MCP Server

MCP server exposing Kagi Search and Summarizer APIs for Claude clients.

## Tools

- **search** - Search the web using Kagi's curated, SEO-resistant index
- **summarize** - Summarize URLs or text (supports PDFs, YouTube, audio)
- **web_fetch_js** - Fetch JavaScript-rendered web content with full browser emulation

### web_fetch_js Capabilities

Renders pages using a headless WebKit browser, enabling access to content that requires JavaScript execution:

- **JS-heavy sites** - SPAs, React/Vue/Angular apps, dynamically loaded content
- **Live app frameworks** - Automatic detection of Gradio and Streamlit apps with accelerated loading (avoids networkidle timeouts)
- **Embedded iframes** - Extracts content from iframes when main page is sparse (e.g., HuggingFace Spaces)
- **Interactive elements** - Returns annotated selectors for ReAct-style interaction chains

**ReAct interaction example:**
```python
# First call: fetch page, observe interactive elements
result = web_fetch_js(url="https://example.com/app")

# Follow-up: interact with discovered elements
result = web_fetch_js(
    url="https://example.com/app",
    actions=[
        {"action": "fill", "selector": "input[name=query]", "value": "search term"},
        {"action": "click", "selector": "button#submit"}
    ]
)
```

## Setup

### API Key

Set your Kagi API key via environment variable or config file:

```bash
# Option 1: Environment variable
export KAGI_API_KEY="your-api-key"

# Option 2: Config file
mkdir -p ~/.config/kagi
echo "your-api-key" > ~/.config/kagi/api_key
```

Get your API key at https://kagi.com/settings?p=api

### Browser Engine (for web_fetch_js)

The `web_fetch_js` tool requires a Playwright browser engine. Install one or both:

```bash
# WebKit (lightweight, preferred when available)
playwright install webkit

# Chromium (broader compatibility, larger download)
playwright install chromium
```

**Browser selection logic:**
1. If `PLAYWRIGHT_BROWSER` env var is set, use that browser
2. If only one browser is installed, use it
3. If multiple browsers available, prefer WebKit (specialized) over Chromium (ubiquitous)

**Override example:**
```bash
# Force Chromium even if WebKit is available
export PLAYWRIGHT_BROWSER=chromium
```

The active browser is shown in tool output: `[Browser: WebKit | ...]`

## Configuration

### Claude Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "kagi": {
      "command": "uv",
      "args": ["--directory", "/path/to/kagi_search_tool", "run", "kagi-mcp"]
    }
  }
}
```

### Claude Desktop (macOS)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kagi": {
      "command": "uv",
      "args": ["--directory", "/path/to/kagi_search_tool", "run", "kagi-mcp", "--profile", "desktop"]
    }
  }
}
```

## Profile Options

The `--profile` argument adjusts tool descriptions for the target client:

| Profile | Target | Built-in tools referenced |
|---------|--------|---------------------------|
| `code` (default) | Claude Code | `WebSearch`, `WebFetch` |
| `desktop` | Claude Desktop | `web_search`, `web_fetch` |

Both profiles position Kagi tools as fallbacks for when built-in tools return poor results.

## Usage

```bash
# Default (Claude Code profile)
kagi-mcp

# Explicit Claude Code profile
kagi-mcp --profile code

# Claude Desktop profile
kagi-mcp --profile desktop

# Show help
kagi-mcp --help
```
