#!/bin/bash
# Glyphh search gate — blocks Grep, Glob, and Bash grep/find until
# glyphh_search has been called at least once in this session.
#
# Installed by `glyphh code init` as a PreToolUse hook.
# Exit 0 = allow, Exit 2 = block (message sent to model as feedback).
#
# Receives JSON on stdin with tool_name and tool_input.

GLYPHH_DIR="${1:-.glyphh}"

# If glyphh_search has already been called, allow everything
[ -f "$GLYPHH_DIR/.search_used" ] && exit 0

# Read hook input from stdin
INPUT=$(cat)

# Extract tool_name without jq (parse JSON with python or grep)
TOOL=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

# For Grep/Glob tools — always block before search
if [ "$TOOL" = "Grep" ] || [ "$TOOL" = "Glob" ]; then
    echo "BLOCKED: Call glyphh_search first. Grep/Glob unlock after glyphh_search has been called." >&2
    exit 2
fi

# For Bash tool — block if the command looks like a file search
if [ "$TOOL" = "Bash" ]; then
    CMD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)
    if echo "$CMD" | grep -qiE '\bgrep\b|\brg\b|\bfind\b|\bfd\b'; then
        echo "BLOCKED: Call glyphh_search first. Bash grep/find/rg unlock after glyphh_search has been called." >&2
        exit 2
    fi
fi

exit 0
