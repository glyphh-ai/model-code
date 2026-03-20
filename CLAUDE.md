# Glyphh Code Intelligence

This project uses Glyphh for codebase search.
The Glyphh index is compiled from every file in this repo.
Follow these rules in every session without exception.


## Tools available

glyphh_search   find files by natural language query
glyphh_related  find files related to a given file
glyphh_stats    index statistics


## Navigation rules

ALWAYS call glyphh_search before reading any file.
ALWAYS call glyphh_related before editing a file.
NEVER scan directories to find relevant code.
NEVER read multiple files speculatively.

Search results include top_tokens and imports for each file.
Use top_tokens to understand what the file is about.
Use imports to understand what it depends on.
Only read the file if top_tokens and imports do not answer the question.
Prefer files with confidence above 0.70.
If the result state is ASK, tell the user the candidates and ask which to use.


## Debugging rules

When investigating a bug or error:
  1. Call glyphh_search with the error type or concept from the stack trace
  2. Check top_tokens and imports from results before reading any file
  3. Read only files with confidence above 0.70
  4. Call glyphh_related on the target file before making any change


## Editing rules

Before editing any file:
  1. Call glyphh_related to understand blast radius
  2. Review top_tokens and imports of related files


## Query guide

Good queries for glyphh_search use specific domain vocabulary:
  auth token validation
  stripe webhook handler
  user profile fetch
  database connection pool
  error boundary component
  payment retry logic
  session expiry check

Poor queries are too generic and will return low-confidence results:
  utils
  helper
  index
  common
  base


## Search result shape

glyphh_search returns:

  state         DONE or ASK
  matches       list of results when state is DONE
    file        relative file path
    confidence  0.0 to 1.0, prefer above 0.70
    top_tokens  dominant concepts in the file
    imports     what the file depends on
    extension   file type
  candidates    list of options when state is ASK

glyphh_related returns:

  state         DONE or ASK
  file          the queried file
  related       list of semantically similar files
    file        relative file path
    similarity  0.0 to 1.0
    top_tokens  dominant concepts
    imports     dependencies
