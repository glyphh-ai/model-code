#!/usr/bin/env bash
#
# Claude Code PostToolUse hook — keep the Glyphh index in sync with git.
#
# Detects three git operations and recompiles accordingly:
#
#   git commit   → incremental compile (changed files in that commit)
#   git pull     → incremental compile (ORIG_HEAD..HEAD diff)
#   git checkout / git switch → full recompile (branch has different files)
#
# Usage in .claude/settings.json:
#
#   "command": "/path/to/post-git-compile.sh /path/to/source/dir"
#
# The first argument is the source directory to index. Any git operation
# inside that directory tree (including child repos and submodules)
# triggers a recompile.
#
# Configuration (environment variables):
#   GLYPHH_RUNTIME_URL   Runtime endpoint (default: http://localhost:8002)
#   GLYPHH_TOKEN         Auth token (auto-resolved from CLI session if unset)
#   GLYPHH_ORG_ID        Org ID (auto-resolved from CLI session if unset)
#   GLYPHH_PYTHON        Python interpreter (default: /opt/homebrew/anaconda3/bin/python)
#   GLYPHH_HOOK_DISABLE  Set to "1" to temporarily disable
#

# Allow disabling without removing the hook
if [ "${GLYPHH_HOOK_DISABLE:-}" = "1" ]; then
    exit 0
fi

# Source directory is the first argument
SOURCE_DIR="${1:?Usage: post-git-compile.sh /path/to/source/dir}"
SOURCE_DIR="$(cd "$SOURCE_DIR" 2>/dev/null && pwd)" || exit 0

# Read the tool input from stdin (JSON with tool_name, tool_input, cwd, etc.)
INPUT="$(cat)"

# Extract the command that was run
COMMAND="$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || true)"

# Detect which git operation happened
MODE=""
if [[ "$COMMAND" == *"git commit"* ]]; then
    MODE="commit"
elif [[ "$COMMAND" == *"git pull"* ]]; then
    MODE="pull"
elif [[ "$COMMAND" == *"git checkout"* ]] || [[ "$COMMAND" == *"git switch"* ]]; then
    # Only trigger on branch switches, not file checkouts
    # git checkout -- file.txt / git checkout HEAD file.txt should not trigger
    if [[ "$COMMAND" == *"git checkout --"* ]] || [[ "$COMMAND" == *"git checkout HEAD "* ]]; then
        exit 0
    fi
    MODE="branch"
elif [[ "$COMMAND" == *"git merge"* ]] || [[ "$COMMAND" == *"git rebase"* ]]; then
    MODE="pull"
fi

# Exit if no relevant git operation detected
if [ -z "$MODE" ]; then
    exit 0
fi

# Determine the repo directory where the git operation ran.
# Commands may cd first: "cd /path && git commit ..."
# Otherwise the command runs in the hook's cwd.
CWD="$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null || true)"
CD_PATH="$(echo "$COMMAND" | sed -n 's|^cd \([^ ;&]*\).*|\1|p')"

if [ -n "$CD_PATH" ] && [ -d "$CD_PATH" ]; then
    GIT_DIR="$(cd "$CD_PATH" && git rev-parse --show-toplevel 2>/dev/null || echo "$CD_PATH")"
elif [ -n "$CWD" ] && [ -d "$CWD" ]; then
    GIT_DIR="$(cd "$CWD" && git rev-parse --show-toplevel 2>/dev/null || echo "$CWD")"
else
    exit 0
fi

# Normalize to absolute path
GIT_DIR="$(cd "$GIT_DIR" 2>/dev/null && pwd)" || exit 0

# Only trigger if the operation is inside SOURCE_DIR
case "$GIT_DIR" in
    "$SOURCE_DIR"|"$SOURCE_DIR/"*)
        ;;
    *)
        exit 0
        ;;
esac

# Locate compile.py relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILE="$SCRIPT_DIR/../compile.py"

if [ ! -f "$COMPILE" ]; then
    exit 0
fi

# Build compile arguments based on the operation
ARGS=("$SOURCE_DIR")

case "$MODE" in
    commit)
        # Incremental: diff HEAD^ → HEAD
        ARGS+=("--incremental")
        if [ "$GIT_DIR" != "$SOURCE_DIR" ]; then
            ARGS+=("--diff-repo" "$GIT_DIR")
        fi
        ;;
    pull)
        # Incremental: diff ORIG_HEAD → HEAD (covers all pulled/merged commits)
        # ORIG_HEAD is set by git pull, git merge, and git rebase
        ORIG_HEAD="$(cd "$GIT_DIR" && git rev-parse ORIG_HEAD 2>/dev/null || true)"
        if [ -n "$ORIG_HEAD" ]; then
            ARGS+=("--diff-range" "${ORIG_HEAD}..HEAD")
        else
            # ORIG_HEAD not available — fall back to full recompile
            echo "[glyphh] pull detected but no ORIG_HEAD — full recompile" >> /tmp/glyphh-compile.log
        fi
        if [ "$GIT_DIR" != "$SOURCE_DIR" ]; then
            ARGS+=("--diff-repo" "$GIT_DIR")
        fi
        ;;
    branch)
        # Full recompile — the file tree may be completely different
        echo "[glyphh] branch switch detected — full recompile" >> /tmp/glyphh-compile.log
        ;;
esac

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

# glyphh SDK lives alongside the source
export PYTHONPATH="$SOURCE_DIR/glyphh-runtime${PYTHONPATH:+:$PYTHONPATH}"

# Run in background — don't block Claude
"$PYTHON" "$COMPILE" "${ARGS[@]}" >> /tmp/glyphh-compile.log 2>&1 &
