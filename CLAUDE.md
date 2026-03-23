# Glyphh Code Intelligence

This project uses Glyphh for semantic codebase analysis.
The Glyphh index is compiled from every file in this repo.
Follow these rules in every session without exception.


## Tools available

glyphh_search   find files by semantic query (concepts, not exact strings)
glyphh_related  find files related to a given file (blast radius)
glyphh_stats    index statistics


## When to use Glyphh vs Grep

Use Grep/Glob for **navigation** — finding where something is defined,
exact string matches, symbol lookups. Grep is faster and cheaper.

Use glyphh_search for **semantic queries** that Grep cannot answer:
  "files related to the payment retry flow"
  "what handles webhook validation"
  "authentication middleware chain"

Use glyphh_related **before editing** any file to understand blast radius.
This returns semantically similar files that may need coordinated changes.
There is no Grep equivalent for this.


## Editing rules

Before editing any file:
  1. Call glyphh_related to understand blast radius
  2. Review top_tokens and imports of related files

After editing:
  A Claude Code PostToolUse hook runs compile.py --incremental in the
  background after every git commit to update the index automatically.
  No manual recompile needed.


## Search result shape

glyphh_search returns:

  state         DONE
  matches       list of results
    file        relative file path
    confidence  0.0 to 1.0, prefer above 0.15
    top_tokens  dominant concepts in the file
    imports     what the file depends on
    extension   file type

glyphh_related returns:

  state         DONE
  file          the queried file
  related       list of semantically similar files
    file        relative file path
    similarity  0.0 to 1.0
    top_tokens  dominant concepts
    imports     dependencies
