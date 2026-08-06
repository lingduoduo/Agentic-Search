---
name: agentic-search
description: Query the Agentic Search knowledge base using the agentic-search command. Use when the user wants to search company documents, ask questions about internal knowledge, query connected data sources, or look up information stored in Agentic Search.
---

# Agentic Search CLI — Agent Tool

`agentic-search` is an agent's interface to the Agentic Search enterprise knowledge platform. It connects to company documents, apps, and people. Use it to answer questions that require internal knowledge — policies, docs, processes, data from connected sources (Confluence, Google Drive, Slack, etc.).

## Prerequisites

### 1. Check if installed

```bash
which agentic-search
```

### 2. Install (if needed)

```bash
pip install -e /path/to/Agentic-Search
```

### 3. Check if configured

If a human has already run `agentic-search query` (which includes first-time setup), the CLI is ready — no additional setup needed. The config file at `~/.config/agentic-search/config.json` (or `$XDG_CONFIG_HOME/agentic-search/config.json` if set) is read automatically.

Environment variables override the config file and can be used as an alternative when no config file exists:

```bash
export AGENTIC_SEARCH_URL="http://localhost:7860"  # default: http://localhost:7860
export AGENTIC_SEARCH_PAT="your-pat"
```

| Variable                        | Required | Description                                                               |
| ------------------------------- | -------- | ------------------------------------------------------------------------- |
| `AGENTIC_SEARCH_URL`            | No       | Server URL (default: `http://localhost:7860`)                             |
| `AGENTIC_SEARCH_PAT`            | Yes      | Personal access token for authentication (unless config file exists)      |
| `AGENTIC_SEARCH_AGENT_ID`       | No       | Default agent/persona ID                                                  |
| `AGENTIC_SEARCH_STREAM_MARKDOWN`| No       | Enable/disable progressive markdown rendering (true/false)                |

If neither a config file nor environment variables are set, tell the user that `agentic-search` needs to be configured and ask them to either:
- Run `agentic-search query` to complete first-time setup interactively, or
- Set `AGENTIC_SEARCH_URL` and `AGENTIC_SEARCH_PAT` environment variables

### 4. Verify configuration

```bash
agentic-search validate-config
```

Exit code 0 on success. Non-zero with a descriptive error on failure (see exit codes below).

## Commands

### Search documents

```bash
agentic-search search "What is our deployment process?"
```

Returns ranked, cited documents from the Agentic Search knowledge base as JSON. Default output is a lean shape: `{"results": [{title, url, source_type, content, updated_at}, ...]}`. Results contain only documents the LLM judged relevant, ordered by relevance; `content` is the full chunk text of each. Use `--raw` for the full API response (adds per-result `citation_id`).

```bash
# Filter by source
agentic-search search --source slack,google_drive "auth migration status"

# Recent results only
agentic-search search --days 30 "recent production incidents"

# Use a specific agent for scoped search
agentic-search search --agent-id 5 "engineering roadmap"

# Full API response for programmatic use
agentic-search search --raw "API documentation" | jq '.results[].title'

# Skip query expansion for exact matching
agentic-search search --no-query-expansion "exact error message text"
```

| Flag                    | Type   | Description                                                      |
| ----------------------- | ------ | ---------------------------------------------------------------- |
| `--source`              | string | Filter by source type (comma-separated: slack,google_drive)      |
| `--days`                | int    | Only return results from the last N days                         |
| `--agent-id`            | int    | Agent ID for scoped search (inherits filters, document sets)     |
| `--raw`                 | bool   | Output full API response (adds per-result citation_id)           |
| `--no-query-expansion`  | bool   | Skip LLM query expansion (faster, less comprehensive)            |
| `--max-output`          | int    | Max bytes to print before truncating (0 to disable, default 50000 for non-TTY, ignored with --raw) |

### Ask a question

```bash
agentic-search ask "What is our company's PTO policy?"
```

Streams an LLM-generated answer as plain text to stdout. Use `search` instead when you need the source documents rather than a synthesized answer. When stdout is not a TTY, output is truncated to 50000 bytes and the full response is saved to a temp file (path printed at the end). Use `--max-output 0` to disable truncation.

```bash
# Use a specific agent
agentic-search ask --agent-id 5 "Summarize our Q4 roadmap"

# Pipe context in with the question
cat error.log | agentic-search ask --prompt "Find the root cause"

# Structured NDJSON output
agentic-search ask --json "List all active API integrations"
```

| Flag           | Type | Description                                                  |
| -------------- | ---- | ------------------------------------------------------------ |
| `--agent-id`   | int  | Agent ID to use (overrides default)                          |
| `--json`       | bool | Output NDJSON stream events instead of plain text (bypasses truncation) |
| `--quiet`      | bool | Buffer output and print once at end (no streaming)           |
| `--prompt`     | str  | Question text (use with piped stdin context)                 |
| `--max-output` | int  | Max bytes to print before truncating (0 to disable, default 50000 for non-TTY) |

### List available agents

```bash
agentic-search agents
agentic-search agents --json
```

Prints a table of agent IDs, names, and descriptions. Use `--json` for structured JSON output. Use agent IDs with `search --agent-id` or `ask --agent-id`.

### Validate configuration

```bash
agentic-search validate-config
```

Checks config exists, PAT is present, server is reachable, and credentials are valid. Use before `search`, `ask`, or `agents` to confirm the CLI is properly set up.

## Output Conventions

- **stdout**: Results only (answer text, agent list, status)
- **stderr**: Progress indicators, warnings, errors
- **Non-TTY**: No ANSI escape codes, no interactive prompts
- **Truncation**: When stdout is not a TTY, `search` and `ask` output is truncated to 50000 bytes. Full response is saved to a temp file whose path is printed. Read the temp file for more.

## Exit Codes

| Code | Name           | Meaning                          |
| ---- | -------------- | -------------------------------- |
| 0    | Success        | Command completed successfully   |
| 1    | General        | Unknown or unclassified error    |
| 2    | BadRequest     | Invalid arguments                |
| 3    | NotConfigured  | Missing config or PAT            |
| 4    | AuthFailure    | Invalid PAT (401/403)            |
| 5    | Unreachable    | Server unreachable               |
| 6    | RateLimited    | Server returned 429              |
| 7    | Timeout        | Request timed out                |
| 8    | ServerError    | Server returned 5xx              |
| 9    | NotAvailable   | Feature/endpoint does not exist  |

## Statelessness

Each invocation is independent. `search` does not create a chat session. `ask` creates a one-shot chat session. There is no way to chain context across multiple invocations — every call starts fresh.

## When to Use

Use `agentic-search search` when:
- You need to find specific documents or gather context for a task
- You want to reason over multiple source documents yourself
- The user asks you to look up or find information in company knowledge
- You need cited, structured results (document IDs, source types, content)

Use `agentic-search ask` when:
- The user wants a direct answer, summarization, or synthesis
- A human-readable response is more useful than raw documents
- You need the LLM to reason across sources and produce an answer

Do NOT use either when:
- The question is about general programming knowledge (use your own knowledge)
- The user is asking about code in the current repository (use grep/read tools)
- The user hasn't mentioned Agentic Search and the question doesn't require internal company data

## Examples

```bash
# Search for documents
agentic-search search "What is our deployment process?"
agentic-search search --source slack "auth migration status"
agentic-search search --raw "API documentation" | jq '.results[].title'

# Ask for an answer
agentic-search ask "What are the steps to deploy to production?"
agentic-search ask --agent-id 3 "What were the action items from last week's standup?"
cat error.log | agentic-search ask --prompt "What does this error mean?"
```
