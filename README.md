# Claude Web Tools

MCP server providing web browsing and content extraction tools for Claude.

## Tools

### Kagi Integration
- **search** - Search the web using Kagi's curated, SEO-resistant index
- **summarize** - Summarize URLs or text (supports PDFs, YouTube, audio)

### Browser Tools
- **WebFetchJS** / **web_fetch_js** - Fetch JavaScript-rendered web content with full browser emulation (name varies by profile)

### web_fetch_js Capabilities

Renders pages using a headless browser, enabling access to content that requires JavaScript execution:

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

### Kagi API Key (for search/summarize tools)

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

The `web_fetch_js` tool requires a Playwright browser engine. Install one or more:

```bash
# WebKit (lightweight, preferred when available)
uv run playwright install webkit

# Chromium (broader compatibility, larger download)
uv run playwright install chromium

# Firefox (alternative option)
uv run playwright install firefox
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
    "claude-web-tools": {
      "command": "uv",
      "args": ["--directory", "/path/to/claude-web-tools", "run", "claude-web-tools"]
    }
  }
}
```

### Claude Desktop (macOS)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "claude-web-tools": {
      "command": "uv",
      "args": ["--directory", "/path/to/claude-web-tools", "run", "claude-web-tools", "--profile", "desktop"]
    }
  }
}
```

## Profile Options

The `--profile` argument adjusts tool names and descriptions for the target client:

| Profile | Target | Tool Naming | Built-in tools referenced |
|---------|--------|-------------|---------------------------|
| `code` (default) | Claude Code | `WebFetchJS` | `WebSearch`, `WebFetch` |
| `desktop` | Claude Desktop | `web_fetch_js` | `web_search`, `web_fetch` |

Tool names align with each client's naming conventions for self-documenting behavior.

## Usage

```bash
# Default (Claude Code profile)
claude-web-tools

# Explicit Claude Code profile
claude-web-tools --profile code

# Claude Desktop profile
claude-web-tools --profile desktop

# Show help
claude-web-tools --help
```
