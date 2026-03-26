# Glyphh Code

File-level codebase intelligence for Claude Code. Encodes every source file in
your repo as an HDC vector. Claude Code queries the index instead of scanning
files.

Same architecture as glyphh-pipedream (3,146 apps) and glyphh-bfcl (#1 on
BFCL V4). No LLM at build time. No LLM at search time. Pure HDC encoding and
cosine search.

Built on [**Glyphh Ada 1.1**](https://www.glyphh.ai/products/runtime) · **[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)** · **License: [AGPL-3.0](LICENSE)**

---

> **Benchmark results** on a 766-file repo (fastmcp), Sonnet 4.6:
>
> ### Blast radius: 1 tool call vs 32
>
> | Metric | With Glyphh | Without Glyphh | Delta |
> |--------|-------------|----------------|-------|
> | **Cost** | $0.10 – $0.17 | $0.21 – $0.23 | **-26% to -50%** |
> | **API time** | 14 – 16s | 58 – 68s | **-72% to -79%** |
> | **Tool calls** | 1 MCP call | 14 – 32 (Explore agent) | **-93% to -97%** |
>
> Same answer. One `glyphh_related` call replaces an Explore subagent that
> greps/globs/reads its way to the same result across 14-32 tool calls.
>
> ### Capabilities with zero grep equivalent
>
> | Tool | What it does | Benchmark |
> |------|-------------|-----------|
> | `glyphh_drift` | Semantic drift score for a file (cosmetic → architectural) | 3/3, $0.08/query |
> | `glyphh_risk` | Aggregate risk profile for a commit or working tree | 2/2, $0.08/query |
>
> **Not a Grep replacement.** Grep wins for navigation and file search — the
> LLM is excellent at grepping its way to any answer. Glyphh adds blast radius
> analysis, drift scoring, and risk profiling — things grep cannot do.
>
> See [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) for full results.

## Prerequisites

- **Python ≥ 3.10**
- **Claude Code CLI** — `npm install -g @anthropic-ai/claude-code`
  (required for `code init` to register the MCP server and configure hooks)


## Quick Start

One install, one command. No Docker, no PostgreSQL, no auth required.

```bash
pip install glyphh-code
```

This installs the Glyphh runtime, CLI, and the Code model as a single package.

Then from your project root:

```bash
glyphh            # enter the Glyphh shell
code init .       # deploy model, compile codebase, configure Claude Code
```

That's it. `code init` handles everything:

1. **Starts a local dev server** (SQLite, no Docker needed)
2. **Deploys the Code model** to the running runtime
3. **Compiles your codebase** into an HDC vector index
4. **Configures Claude Code** — adds MCP server, search gate hooks, permissions

Restart Claude Code to activate. In VS Code: `Cmd+Shift+P` → "Claude Code:
Restart". In the CLI: exit and re-enter the session.

Verify the connection with `/mcp` — you should see `glyphh_search`,
`glyphh_related`, `glyphh_drift`, `glyphh_risk`, and `glyphh_stats` listed
as available tools.


## Using PostgreSQL + pgvector (optional)

SQLite works out of the box for local development. For larger codebases or
production use, you can use PostgreSQL with pgvector for faster similarity
search via HNSW indexing.

```bash
# Set DATABASE_URL before starting the shell
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/glyphh

glyphh
code init .
```

Or use the built-in Docker setup:

```bash
glyphh
docker init       # generates docker-compose.yml + init.sql
exit

docker compose up -d --wait
glyphh
code init .
```

The runtime auto-detects the backend from `DATABASE_URL` — no configuration
changes needed. SQLite uses Python cosine similarity; PostgreSQL uses native
pgvector `<=>` with HNSW indexing.


## Shell Commands

After `pip install glyphh-code`, the `code` command is available inside the
Glyphh shell:

| Command | Description |
|---------|-------------|
| `code init [path]` | Full setup: start server, deploy, compile, configure Claude Code |
| `code compile [path]` | Recompile the index (full or incremental) |
| `code status` | Show current status (server, files indexed, MCP URL) |
| `code stop` | Stop the dev server |

The shell also has all standard runtime commands (`dev`, `model`, `auth`,
`config`, etc.). Type `help` for the full list.


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
             → encode into HDC vector → store vector + metadata
             → supports pgvector (HNSW) or SQLite (Python cosine)

Runtime:     NL query → encode with same pipeline
             → cosine search against index
             → return file path + top tokens + imports
             → Claude reads one file, acts
```

No LLM at build time. No LLM at runtime for search.


## Encoder

Three-layer HDC encoder at 2,000 dimensions:

| Layer | Weight | Signal |
|-------|--------|--------|
| **path** | 0.30 | File path tokens (BoW): `src/services/user_service.py` → `src services user service py` |
| **symbols** | 0.50 | AST-extracted definitions (class/function names via tree-sitter) |
| **content** | 0.20 | Identifiers (1.0) + imports (0.8) as BoW |

The symbols layer encodes what a file **defines**, not what it uses. This
naturally separates source files from their tests — `auth.py` defines
`AuthMiddleware` while `test_auth.py` defines `test_check_auth`.

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
understand blast radius. This is Glyphh's primary differentiator — one call
replaces 14-32 grep/glob cycles.

```json
{"tool": "glyphh_related", "arguments": {"file_path": "src/services/auth.py", "top_k": 5}}
```

### glyphh_drift

Compute semantic drift for a file since the last index build. Returns a score
(0.0–1.0) and label (cosmetic, moderate, significant, architectural).

```json
{"tool": "glyphh_drift", "arguments": {"file_path": "src/services/auth.py"}}
```

### glyphh_risk

Aggregate risk profile for all changed files in the working tree or a git ref.
Returns per-file drift scores, overall risk label, and hot files.

```json
{"tool": "glyphh_risk", "arguments": {}}
{"tool": "glyphh_risk", "arguments": {"git_ref": "HEAD~1"}}
```

### glyphh_stats

Index statistics: total files, extension breakdown.


## Incremental compile

The index is updated automatically after every `git commit` via the Claude Code
PostToolUse hook configured by `code init`.

For manual recompilation:

```bash
glyphh
code compile .                    # full recompile
code compile /path/to/repo        # compile a different repo
```


## File support

Indexes: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`, `.cpp`, `.c`, `.h`,
`.go`, `.rs`, `.rb`, `.cs`, `.swift`, `.sql`, `.graphql`, `.yaml`, `.json`,
`.sh`, `.css`, `.html`, `.svelte`, `.vue`, `.md`, `.proto`, `.tf`, and more.

Skips: `.git`, `node_modules`, `__pycache__`, `dist`, `build`, `vendor`,
`target`, and other build/cache directories.

Max file size: 500 KB. Binary files auto-skipped.


## What `code init` configures

Running `code init .` in the Glyphh shell sets up the following in your
project:

- **MCP server** — `claude mcp add --transport http glyphh <url>`
- **`.claude/settings.json`** — hooks and permissions:
  - `mcp__glyphh__*` permission (no MCP prompts)
  - PreToolUse hook: gates Grep/Glob/Bash until `glyphh_search` has been called
  - PostToolUse hook: runs incremental compile after `git commit`
- **`.gitignore`** — adds `.glyphh/` entry
- **CLAUDE.md migration** — removes previously injected Glyphh sections (if any)


## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite (`~/.glyphh/local.db`) | Database connection string |
| `GLYPHH_RUNTIME_URL` | `http://localhost:8002` | Runtime endpoint |
| `GLYPHH_TOKEN` | Auto-resolved from CLI session | Auth token |
| `GLYPHH_ORG_ID` | Auto-resolved from CLI session | Org ID |
| `GLYPHH_HOOK_DISABLE` | — | Set to `1` to temporarily disable hooks |


## Tests

```bash
pip install glyphh-code[dev]
pytest tests/ -v
```
