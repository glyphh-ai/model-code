# Glyphh Code Intelligence

This project uses Glyphh for semantic codebase analysis.
The Glyphh index is compiled from every file in this repo.
Follow these rules in every session without exception.


## Tools available

glyphh_search   find files by semantic query (concepts, not exact strings)
glyphh_related  find files related to a given file (blast radius)
glyphh_drift    semantic drift score for a file since last index build
glyphh_risk     risk profile for changed files (aggregates drift scores)
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


## Deploy and merge rules

Before suggesting a deploy or approving a merge:
  1. Call glyphh_risk to score the current commit
  2. If risk_label is significant or architectural, flag for human review
  3. If hot_files is non-empty, call glyphh_drift on each hot file
  4. Report the risk profile to the user before proceeding


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

glyphh_drift returns:

  state         DONE
  file          the queried file
  drift_score   0.0 to 1.0
  drift_label   cosmetic, moderate, significant, or architectural

glyphh_risk returns:

  state         DONE
  files         per-file drift scores
  max_drift     highest single-file drift
  mean_drift    average across changed files
  risk_label    cosmetic, moderate, significant, or architectural
  hot_files     files above the moderate threshold
