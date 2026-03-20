# Glyphh Code

File-level codebase intelligence for Claude Code. Encodes every source file in
your repo as an HDC vector. Claude Code queries the index instead of scanning
files.

Same architecture as glyphh-pipedream (3,146 apps) and glyphh-bfcl (#1 on
BFCL V4). No LLM at build time. No LLM at search time. Pure HDC encoding and
cosine search.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

---

## Getting Started

### 1. Install the Glyphh CLI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with runtime dependencies (includes FastAPI, SQLAlchemy, pgvector)
pip install 'glyphh[runtime]'
```

### 2. Clone and start the model

This model requires PostgreSQL + pgvector for similarity search.

```bash
git clone https://github.com/glyphh-ai/model-code.git
cd model-code

# Start the Glyphh shell (prompts login on first run)
glyphh

# Inside the shell:
# glyphh> docker init       # generates docker-compose.yml + init.sql
# glyphh> exit

# Start PostgreSQL + pgvector and the Glyphh runtime
docker compose up -d --wait
```

This starts:
- **PostgreSQL 16 + pgvector** on port 5432 (with HNSW indexing)
- **Glyphh Runtime** on port 8002

Swagger docs available at `http://localhost:8002/docs` in local mode.

### 3. Deploy the model

```bash
glyphh
# glyphh> model deploy .     # deploy code model to runtime
```

### 4. Compile your codebase

```bash
# Full compile (all indexable files)
python compile.py /path/to/your/repo --runtime-url http://localhost:8002

# Incremental (changed files since last commit)
python compile.py /path/to/your/repo --incremental

# Dry run (show what would be indexed)
python compile.py /path/to/your/repo --dry-run
```

### 5. Connect Claude Code

Add the MCP server using the Claude Code CLI:

```bash
claude mcp add --transport http glyphh-code http://localhost:8002/{org_id}/code/mcp
```

To find your org ID, run `glyphh auth status` in the Glyphh shell:

```bash
glyphh
# glyphh> auth status
#   org_id: your-org-id-here
```

In local mode the org ID is `local-dev-org`:

```bash
claude mcp add --transport http glyphh-code http://localhost:8002/local-dev-org/code/mcp
```

Restart Claude Code to pick up the MCP config. In VS Code: `Cmd+Shift+P` →
"Claude Code: Restart". In the CLI: exit and re-enter the session.

Verify the connection with `/mcp` — you should see `glyphh_search`,
`glyphh_related`, and `glyphh_stats` listed as available tools.

### 6. Add CLAUDE.md (recommended)

Copy the included `CLAUDE.md` into your project root:

```bash
cp CLAUDE.md /path/to/your/project/CLAUDE.md
```

Claude Code loads this file automatically at the start of every conversation.
It teaches Claude Code to always search the Glyphh index before reading files,
check blast radius before editing, and use `top_tokens` and `imports` from
search results to avoid unnecessary file reads.

Without it, Claude Code will still have the MCP tools available but will fall
back to its default file scanning behavior.

---

## What it does

Compiles your codebase into a vector index. Exposes it to Claude Code via MCP.

**Without Glyphh:**
Claude reads project structure, scans likely files, reads module, reads tests.
~6,000 tokens before first useful output.

**With Glyphh:**
Claude calls `glyphh_search("auth token validation")`.
Returns: file path, confidence, top concepts, imports, related files.
Claude reads one file and acts.
~400 tokens before first useful output.

The index stores not just the vector but the token vocabulary of every file.
Search results return enough context that Claude often does not need to read
the file at all. When it does read, it already knows exactly what to look for.


## Architecture

Same paradigm as all Glyphh models. The file is the exemplar.

```
Build time:  read file → tokenize path + identifiers + imports
             → encode into HDC vector → store vector + metadata in pgvector

Runtime:     NL query → encode with same pipeline
             → cosine search against index
             → return file path + top tokens + imports
             → Claude reads one file, acts
```

No LLM at build time. No LLM at runtime for search.


## Encoder

Two-layer HDC encoder at 2,000 dimensions (pgvector HNSW compatible):

| Layer | Weight | Signal |
|-------|--------|--------|
| **path** | 0.30 | File path tokens (BoW): `src/services/user_service.py` → `src services user service py` |
| **content** | 0.70 | Source file vocabulary |
| ↳ identifiers | 1.0 | All tokens from file content. camelCase/snake_case split before encoding |
| ↳ imports | 0.8 | Import/require/include targets. Strong cross-file dependency signal |

Metadata stored per file (not encoded, returned at search time):
- `top_tokens`: 20 most frequent meaningful tokens
- `imports`: list of imported module/package names
- `extension`: file type
- `file_size`: bytes


## MCP Tools

Exposed through the runtime's model-specific MCP tool system:

### glyphh_search

Find files by natural language query. Returns ranked matches with confidence
scores, top tokens, and import lists.

```json
{"tool": "glyphh_search", "arguments": {"query": "auth token validation", "top_k": 5}}
```

Confidence gate: below threshold returns ASK with candidates, never silent
wrong routing.

### glyphh_related

Find files semantically related to a given file. Use before editing to
understand blast radius.

```json
{"tool": "glyphh_related", "arguments": {"file_path": "src/services/auth.py", "top_k": 5}}
```

### glyphh_stats

Index statistics: total files, extension breakdown.


## Drift scoring

The `drift.py` module computes semantic drift between file versions:

| Drift | Label | Meaning |
|-------|-------|---------|
| 0.00–0.10 | cosmetic | Formatting, comments, rename |
| 0.10–0.30 | moderate | Logic update, new function |
| 0.30–0.60 | significant | Behavioral change, new dependency |
| 0.60–1.00 | architectural | Rewrite, interface change |


## Incremental compile

```bash
# Recompile only files changed in the last commit
python compile.py . --incremental

# Recompile files changed in a specific commit
python compile.py . --diff abc123
```

### Post-commit hook

A ready-to-use hook is included in `hooks/post-commit`. It runs
`compile.py --incremental` in the background after every commit so the
index stays up to date automatically.

Install it in any repo you've compiled:

```bash
# Copy (one-time)
cp /path/to/model-code/hooks/post-commit /path/to/your/repo/.git/hooks/post-commit

# Or symlink (auto-updates)
ln -sf /path/to/model-code/hooks/post-commit /path/to/your/repo/.git/hooks/post-commit
```

If `compile.py` isn't in the repo root, point the hook to it:

```bash
export GLYPHH_COMPILE_PATH=/path/to/model-code/compile.py
```

The hook runs in the background and won't slow down your commits.
Disable temporarily with `GLYPHH_HOOK_DISABLE=1`.


## File support

Indexes: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`, `.cpp`, `.c`, `.h`,
`.go`, `.rs`, `.rb`, `.cs`, `.swift`, `.sql`, `.graphql`, `.yaml`, `.json`,
`.sh`, `.css`, `.html`, `.svelte`, `.vue`, `.md`, `.proto`, `.tf`, and more.

Skips: `.git`, `node_modules`, `__pycache__`, `dist`, `build`, `vendor`,
`target`, and other build/cache directories.

Max file size: 500 KB. Binary files auto-skipped.


## Disable MCP permission prompts

By default Claude Code prompts for permission each time it calls an MCP tool.
To allow Glyphh tools silently, add them to `.claude/settings.json` in your
project:

```json
{
  "permissions": {
    "allow": [
      "mcp__glyphh-code__glyphh_search",
      "mcp__glyphh-code__glyphh_related",
      "mcp__glyphh-code__glyphh_drift",
      "mcp__glyphh-code__glyphh_risk"
    ]
  }
}
```

Or use a wildcard to allow all tools from the Glyphh server:

```json
{
  "permissions": {
    "allow": [
      "mcp__glyphh-code__*"
    ]
  }
}
```

The first matching rule wins — Glyphh tools run silently while everything else
still prompts.


## Enforce glyphh_search over Grep/Glob

Claude Code defaults to using Grep and Glob for file search — bypassing the
Glyphh index entirely. The included CLAUDE.md rules tell Claude to use
`glyphh_search` first, but Claude doesn't always follow them.

A Claude Code **PreToolUse hook** can enforce this by blocking Grep and Glob
calls with a message redirecting Claude to `glyphh_search`.

Add this to `.claude/settings.json` in your project:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/model-code/hooks/enforce-glyphh-search.sh"
          }
        ]
      }
    ]
  }
}
```

Replace `/path/to/model-code` with wherever you cloned this repo.

When Claude tries to call Grep or Glob, the hook blocks the call and tells
Claude to use `glyphh_search` instead. Claude will then retry with the
Glyphh index.

To temporarily disable the hook, remove or comment out the `PreToolUse`
entry in settings.json.


## Tests

```bash
cd glyphh-models/code
PYTHONPATH=../../glyphh-runtime pytest tests/ -v
```
