#!/usr/bin/env bash
#
# Claude Code PreToolUse hook — enforce glyphh_search over Grep/Glob.
#
# Blocks Grep and Glob tool calls with a message telling Claude to use
# glyphh_search instead. This prevents Claude from bypassing the Glyphh
# index and scanning the filesystem directly.
#
# Install by adding to .claude/settings.json:
#
#   {
#     "hooks": {
#       "PreToolUse": [
#         {
#           "matcher": "Grep|Glob",
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/model-code/hooks/enforce-glyphh-search.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }
#

# The hook receives tool info on stdin as JSON.
# Exit 2 = block the tool call with a message on stderr.

echo "BLOCKED: Use glyphh_search instead of Grep/Glob. The Glyphh index is faster and more accurate. Only fall back to Grep/Glob if glyphh_search returns no results above 0.50." >&2
exit 2
