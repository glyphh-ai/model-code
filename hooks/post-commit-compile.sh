#!/usr/bin/env bash
#
# Claude Code PostToolUse hook — incremental index update after git commit.
#
# Runs compile.py --incremental in the background after Claude commits,
# keeping the Glyphh index up to date automatically.
#
# Install by adding to .claude/settings.json:
#
#   {
#     "hooks": {
#       "PostToolUse": [
#         {
#           "matcher": "Bash",
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/model-code/hooks/post-commit-compile.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }
#
# Configuration (environment variables):
#   GLYPHH_RUNTIME_URL   Runtime endpoint (default: http://localhost:8002)
#   GLYPHH_TOKEN         Auth token (auto-resolved from CLI session if unset)
#   GLYPHH_ORG_ID        Org ID (auto-resolved from CLI session if unset)
#   GLYPHH_HOOK_DISABLE  Set to "1" to temporarily disable
#

# Allow disabling without removing the hook
if [ "${GLYPHH_HOOK_DISABLE:-}" = "1" ]; then
    exit 0
fi

# Read the tool input from stdin (JSON with tool_name, tool_input, etc.)
INPUT="$(cat)"

# Only trigger on Bash commands that contain "git commit"
COMMAND="$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || true)"

if [[ "$COMMAND" != *"git commit"* ]]; then
    exit 0
fi

# The hook always targets glyphh-master as the indexed repo,
# regardless of which subdirectory the commit happened in.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILE="$SCRIPT_DIR/../compile.py"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ ! -f "$COMPILE" ]; then
    exit 0
fi

# Build args
ARGS=("$REPO_ROOT" "--incremental")

if [ -n "${GLYPHH_RUNTIME_URL:-}" ]; then
    ARGS+=("--runtime-url" "$GLYPHH_RUNTIME_URL")
fi

if [ -n "${GLYPHH_TOKEN:-}" ]; then
    ARGS+=("--token" "$GLYPHH_TOKEN")
fi

if [ -n "${GLYPHH_ORG_ID:-}" ]; then
    ARGS+=("--org-id" "$GLYPHH_ORG_ID")
fi

# Use anaconda python (has requests) — system python3 may not
PYTHON="${GLYPHH_PYTHON:-/opt/homebrew/anaconda3/bin/python}"

# Run in background — don't block Claude
"$PYTHON" "$COMPILE" "${ARGS[@]}" >> /tmp/glyphh-compile.log 2>&1 &
