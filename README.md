# Kagi MCP Server

MCP server exposing Kagi Search and Summarizer APIs for Claude clients.

## Tools

- **search** - Search the web using Kagi's curated, SEO-resistant index
- **summarize** - Summarize URLs or text (supports PDFs, YouTube, audio)

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
