"""
Encoder for the Glyphh Code model — file-level codebase intelligence.

Exports:
    ENCODER_CONFIG     — Two-layer HDC encoder (path + content)
    encode_query       — NL text → Concept dict for similarity search
    entry_to_record    — Raw file record → build record for runtime listener
    file_to_record     — Source file path → record dict (compile-time)
    MCP_TOOLS          — Model-specific MCP tool schemas
    handle_mcp_tool    — Async handler for model-specific MCP calls

Architecture:
    Path layer (0.30):  file path segments as BoW
    Content layer (0.70): identifiers (1.0) + imports (0.8) as BoW
    Dimension: 2000 (pgvector HNSW compatible)

    At compile time: walk repo, tokenize each file, POST records to runtime
    At query time: same two-layer encoding applied to NL query text,
    cosine similarity against file index returns ranked file list.

    Confidence gate: threshold + gap analysis. Below threshold returns ASK
    with candidates, never silent wrong routing.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment, TemporalConfig


# ---------------------------------------------------------------------------
# Encoder config
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=True,
    temporal_config=TemporalConfig(signal_type="auto"),
    temporal_source="auto",
    layers=[
        Layer(
            name="path",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="location",
                    roles=[
                        Role(
                            name="path_tokens",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                            key_part=True,
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="content",
            similarity_weight=0.70,
            segments=[
                Segment(
                    name="vocabulary",
                    roles=[
                        Role(
                            name="identifiers",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="imports",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEXABLE_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".cs", ".swift",
    ".sql", ".graphql", ".gql",
    ".yaml", ".yml", ".toml", ".json",
    ".sh", ".bash", ".zsh",
    ".css", ".scss", ".less",
    ".html", ".svelte", ".vue",
    ".md", ".rst", ".txt",
    ".proto", ".thrift",
    ".tf", ".hcl",
    ".dockerfile",
})

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "dist", "build", ".next", "coverage",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".tox", "egg-info", ".eggs", ".cache",
    "vendor", "target", "out", "bin", "obj",
})

MAX_FILE_BYTES = 500_000

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "in", "of", "to", "for",
    "and", "or", "with", "that", "this", "where", "how",
    "what", "which", "find", "show", "get", "me", "us",
    "it", "be", "on", "at", "by", "do", "if", "my",
    "we", "so", "up", "are", "but", "not", "all", "can",
    "had", "her", "was", "one", "our", "has", "no",
    # Code noise
    "code", "file", "files", "function", "class", "method",
    "def", "var", "let", "const", "return", "import", "from",
    "true", "false", "none", "null", "self", "cls",
    "new", "try", "catch", "throw", "async", "await",
    "public", "private", "protected", "static", "void",
    "int", "str", "bool", "float", "string", "type",
})


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _split_camel(text: str) -> str:
    """Split camelCase/PascalCase: sendSlackMessage → send Slack Message."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def _split_snake(text: str) -> str:
    """Split snake_case and kebab-case: user_service → user service."""
    return text.replace("_", " ").replace("-", " ")


def _tokenize(text: str) -> str:
    """Normalize text to lowercase space-separated tokens."""
    text = _split_camel(text)
    text = _split_snake(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _extract_path_tokens(file_path: str) -> str:
    """Tokenize a file path: src/services/user_service.py → src services user service py."""
    parts = re.split(r"[/\\]", file_path)
    return " ".join(_tokenize(p) for p in parts if p)


def _extract_imports(content: str) -> str:
    """Extract import targets from source code across languages."""
    patterns = [
        r"(?:from|import)\s+([\w\.]+)",                    # Python
        r"(?:import|require)\s*(?:\(?\s*['\"])([\w\-\./@]+)",  # JS/TS
        r"#include\s*[<\"]([\w\./]+)",                     # C/C++
        r"use\s+([\w:]+)",                                 # Rust/Go
    ]
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, content):
            matches.append(_tokenize(m.group(1)))
    return " ".join(matches)


def _extract_identifiers(content: str) -> str:
    """Extract meaningful identifiers from source code.

    Strips string literals, comments, and short tokens.
    Splits camelCase and snake_case before tokenizing.
    """
    # Strip string literals
    content = re.sub(r'["\'].*?["\']', " ", content, flags=re.DOTALL)
    # Strip single-line comments
    content = re.sub(r"//.*$", " ", content, flags=re.MULTILINE)
    content = re.sub(r"#.*$", " ", content, flags=re.MULTILINE)
    # Strip block comments
    content = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    # Extract identifiers
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", content)
    words = [w for w in words if len(w) > 1]
    return " ".join(_tokenize(w) for w in words)


def _top_tokens(identifiers: str, n: int = 20) -> list[str]:
    """Return the top N most frequent meaningful tokens."""
    counts = Counter(identifiers.split())
    return [
        tok
        for tok, _ in counts.most_common(n * 2)
        if tok not in _STOP_WORDS and len(tok) > 2
    ][:n]


# ---------------------------------------------------------------------------
# Structural query analysis
# ---------------------------------------------------------------------------

def _analyze_query(query: str, words: list[str]) -> dict:
    """Infer query intent from structure, not domain assumptions.

    Detects three intent types based on query-intrinsic signals:
      structural — path separators, extensions, camelCase, snake_case
      import     — relational language (import, require, depend, ...)
      concept    — plain natural language (the common case)
    """
    # Path signal — structural indicators only
    has_slash = "/" in query or "\\" in query
    has_extension = bool(re.search(r"\.\w{2,4}$", query.strip()))
    has_camel = bool(re.search(r"[a-z][A-Z]", query))
    has_snake = "_" in query

    is_structural = has_slash or has_extension or has_camel or has_snake

    # Import signal — relational language
    import_words = {
        "import", "require", "depend", "use", "using",
        "from", "package", "library", "module",
    }
    is_import = bool(import_words & set(words))

    return {
        "is_structural": is_structural,
        "is_import": is_import,
        "is_concept": not is_structural and not is_import,
    }


# ---------------------------------------------------------------------------
# encode_query — NL query → concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert NL query to Concept dict for file index search.

    Uses structural query analysis to route attributes:
      structural → path + identifiers (skip imports)
      import     → identifiers + imports (skip path)
      concept    → identifiers + imports (skip path)

    This ensures the 0.30/0.70 path/content weight split works correctly:
    path weight only activates when the query looks like a path or identifier.
    """
    tokens = _tokenize(query)
    words = [w for w in tokens.split() if w not in _STOP_WORDS and len(w) > 1]
    clean = " ".join(words)

    intent = _analyze_query(query, words)

    if intent["is_structural"]:
        path_tokens = clean
        identifiers = clean
        imports = ""
    elif intent["is_import"]:
        path_tokens = ""
        identifiers = clean
        imports = clean
    else:
        path_tokens = ""
        identifiers = clean
        imports = clean

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08x}",
        "attributes": {
            "path_tokens": path_tokens,
            "identifiers": identifiers,
            "imports": imports,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — listener record → build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a file record into a build record for encoding.

    Accepts records from either:
      - file_to_record() output (has concept_text, attributes, metadata)
      - Raw JSONL entries (has file_path, path_tokens, identifiers, imports)

    Returns: {"concept_text": str, "attributes": dict, "metadata": dict}
    """
    # Already in the right shape (from file_to_record)
    if "concept_text" in entry and "attributes" in entry:
        return entry

    # Raw JSONL format
    file_path = entry.get("file_path", entry.get("concept_text", "unknown"))
    return {
        "concept_text": file_path,
        "attributes": {
            "path_tokens": entry.get("path_tokens", _extract_path_tokens(file_path)),
            "identifiers": entry.get("identifiers", ""),
            "imports": entry.get("imports", ""),
        },
        "metadata": {
            "file_path": file_path,
            "extension": entry.get("extension", Path(file_path).suffix),
            "file_size": entry.get("file_size", 0),
            "top_tokens": entry.get("top_tokens", []),
            "imports": entry.get("import_list", []),
        },
    }


# ---------------------------------------------------------------------------
# file_to_record — source file → record dict (compile-time)
# ---------------------------------------------------------------------------

def file_to_record(file_path: str, repo_root: str = ".") -> dict | None:
    """Convert a source file to a Glyphh build record.

    Returns None for binary files, oversized files, or skipped types.
    Returns dict with concept_text, attributes, and metadata.
    Metadata includes top_tokens and imports for rich search results.
    """
    path = Path(file_path)

    if path.suffix not in INDEXABLE_EXTENSIONS:
        return None
    if not path.exists():
        return None
    if path.stat().st_size > MAX_FILE_BYTES:
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    try:
        rel_path = str(path.relative_to(repo_root))
    except ValueError:
        rel_path = str(path)

    path_tokens = _extract_path_tokens(rel_path)
    imports = _extract_imports(content)
    identifiers = _extract_identifiers(content)
    top_tokens = _top_tokens(identifiers)

    # Use top_tokens for encoding instead of full identifiers.
    # Full identifiers can have 200+ tokens which dilutes each token's
    # contribution to ~1/sqrt(N) in the HDC vector. Top 20 tokens keep
    # the density comparable to query vectors (3-5 tokens), making
    # cosine similarity discriminative.
    top_tokens_str = " ".join(top_tokens)

    return {
        "concept_text": rel_path,
        "attributes": {
            "path_tokens": path_tokens,
            "identifiers": top_tokens_str,
            "imports": imports,
        },
        "metadata": {
            "file_path": rel_path,
            "extension": path.suffix,
            "file_size": path.stat().st_size,
            "top_tokens": top_tokens,
            "imports": [t for t in imports.split() if len(t) > 2],
        },
    }


# ---------------------------------------------------------------------------
# MCP Tools — exposed through the runtime's MCP server
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "glyphh_search",
        "description": (
            "Find files in the codebase by natural language query. "
            "ALWAYS call this before reading any file or listing directories. "
            "Returns file paths with confidence scores, top concepts, imports, "
            "and related files. Use top_tokens to understand file content "
            "without reading it. Use imports to understand dependencies. "
            "Only read a file if top_tokens and imports do not answer the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you are looking for",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "glyphh_related",
        "description": (
            "Find files semantically related to a given file path. "
            "ALWAYS call this before editing a file to understand blast radius. "
            "Returns files that share vocabulary, imports, or domain concepts. "
            "Includes top_tokens and imports for each related file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to find related files for",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of related files to return",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "glyphh_stats",
        "description": (
            "Get statistics about the indexed codebase. "
            "Returns total files indexed, file type breakdown, "
            "and last compile time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# MCP tool handler
# ---------------------------------------------------------------------------

async def handle_mcp_tool(tool_name: str, arguments: dict, context: dict) -> dict:
    """Handle model-specific MCP tool calls.

    Dispatches to per-tool handlers. Each handler uses the runtime's
    session_factory to query pgvector directly.

    Context keys:
        org_id, model_id, encoder, encode_query_fn,
        similarity_calculator, model_manager, session_factory
    """
    handlers = {
        "glyphh_search": _handle_search,
        "glyphh_related": _handle_related,
        "glyphh_stats": _handle_stats,
    }

    handler = handlers.get(tool_name)
    if handler is None:
        return _mcp_text(f"Unknown tool: {tool_name}")

    try:
        return await handler(arguments, context)
    except Exception as e:
        return _mcp_text(f"Error in {tool_name}: {e}", is_error=True)


def _mcp_text(text: str, is_error: bool = False) -> dict:
    """Build an MCP response with a single text content block."""
    return {
        "content": [{"type": "text", "text": text}],
        "is_error": is_error,
    }


def _mcp_json(data: dict, is_error: bool = False) -> dict:
    """Build an MCP response with JSON-serialized text content."""
    return {
        "content": [{"type": "text", "text": json.dumps(data, default=str)}],
        "is_error": is_error,
    }


async def _pgvector_search(
    session_factory: Any,
    org_id: str,
    model_id: str,
    query_vector,
    top_k: int = 5,
    exclude_key: str | None = None,
) -> list[dict]:
    """Run cosine similarity search against pgvector.

    Returns list of dicts with concept_text, metadata, score.
    """
    import numpy as np
    from sqlalchemy import text

    vec = np.asarray(query_vector, dtype=float)
    vec_str = "[" + ",".join(f"{v:.1f}" for v in vec) + "]"

    where = "org_id = :org_id AND model_id = :model_id"
    params: dict[str, Any] = {
        "org_id": org_id,
        "model_id": model_id,
        "vec": vec_str,
        "top_k": top_k,
    }

    if exclude_key:
        where += " AND concept_text != :exclude_key"
        params["exclude_key"] = exclude_key

    async with session_factory() as session:
        result = await session.execute(
            text(
                f"SELECT concept_text, metadata, "
                f"1 - (embedding <=> CAST(:vec AS vector)) AS score "
                f"FROM glyphs "
                f"WHERE {where} "
                f"ORDER BY embedding <=> CAST(:vec AS vector) "
                f"LIMIT :top_k"
            ),
            params,
        )
        rows = result.fetchall()

    results = []
    for row in rows:
        meta = row.metadata if isinstance(row.metadata, dict) else {}
        results.append({
            "concept_text": row.concept_text,
            "metadata": meta,
            "score": float(row.score),
        })
    return results


def _format_match(row: dict) -> dict:
    """Format a pgvector result row into a search match.

    The runtime stores the full record as glyph metadata. When records
    are sent in hierarchical format, the model's metadata dict is nested
    under the 'metadata' key. Handle both flat and nested layouts.
    """
    meta = row.get("metadata", {})
    # The model's metadata may be nested under 'metadata' key
    inner = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
    return {
        "file": inner.get("file_path") or meta.get("file_path") or row["concept_text"],
        "confidence": round(row["score"], 3),
        "top_tokens": inner.get("top_tokens") or meta.get("top_tokens", []),
        "imports": inner.get("imports") or meta.get("imports", []),
        "extension": inner.get("extension") or meta.get("extension", ""),
    }


async def _handle_search(arguments: dict, context: dict) -> dict:
    """Find files by NL query with confidence gating."""
    from glyphh.core.types import Concept

    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    if not query.strip():
        return _mcp_text("Query is empty.")

    org_id = context["org_id"]
    model_id = context["model_id"]
    encoder = context["encoder"]
    encode_fn = context["encode_query_fn"]

    # Encode query
    concept_dict = encode_fn(query)
    attrs = concept_dict.get("attributes", concept_dict)
    name = concept_dict.get("name", "query")
    concept = Concept(name=name, attributes=attrs)
    query_glyph = encoder.encode(concept)

    # Search pgvector
    rows = await _pgvector_search(
        context["session_factory"],
        org_id,
        model_id,
        query_glyph.global_cortex.data,
        top_k=top_k,
    )

    if not rows:
        return _mcp_json({
            "state": "ASK",
            "message": "No files indexed. Run: glyphh-code compile .",
        })

    matches = [_format_match(r) for r in rows]
    top_score = matches[0]["confidence"]
    gap = top_score - matches[1]["confidence"] if len(matches) > 1 else 1.0

    if top_score < 0.45 or gap < 0.03:
        return _mcp_json({
            "state": "ASK",
            "candidates": matches[:3],
            "message": "Multiple similar files found. Which did you mean?",
        })

    return _mcp_json({
        "state": "DONE",
        "matches": matches,
    })


async def _handle_related(arguments: dict, context: dict) -> dict:
    """Find files semantically related to a given file."""
    from sqlalchemy import text as sql_text

    file_path = arguments.get("file_path", "")
    top_k = arguments.get("top_k", 5)
    if not file_path.strip():
        return _mcp_text("file_path is required.")

    org_id = context["org_id"]
    model_id = context["model_id"]

    # Get the file's stored vector.
    # concept_text may be a mangled key_part value (tokenized path with
    # underscores) so also search metadata->>'file_path' for the real path.
    async with context["session_factory"]() as session:
        result = await session.execute(
            sql_text(
                "SELECT concept_text, embedding, metadata FROM glyphs "
                "WHERE org_id = :org_id AND model_id = :model_id "
                "AND (concept_text = :file_path "
                "     OR metadata->'metadata'->>'file_path' = :file_path) "
                "LIMIT 1"
            ),
            {"org_id": org_id, "model_id": model_id, "file_path": file_path},
        )
        row = result.fetchone()

    if row is None:
        return _mcp_json({
            "state": "ASK",
            "message": f"File not in index: {file_path}. Run: glyphh-code compile .",
        })

    # Search for similar files, excluding the source file
    stored_key = row.concept_text
    rows = await _pgvector_search(
        context["session_factory"],
        org_id,
        model_id,
        row.embedding,
        top_k=top_k + 1,
        exclude_key=stored_key,
    )

    related = [_format_match(r) for r in rows[:top_k]]
    source_meta = row.metadata if isinstance(row.metadata, dict) else {}
    source_inner = source_meta.get("metadata", {}) if isinstance(source_meta.get("metadata"), dict) else {}

    return _mcp_json({
        "state": "DONE",
        "file": file_path,
        "top_tokens": source_inner.get("top_tokens") or source_meta.get("top_tokens", []),
        "imports": source_inner.get("imports") or source_meta.get("imports", []),
        "related": related,
    })


async def _handle_stats(arguments: dict, context: dict) -> dict:
    """Return index statistics."""
    from sqlalchemy import text as sql_text

    org_id = context["org_id"]
    model_id = context["model_id"]

    async with context["session_factory"]() as session:
        # Total files
        total = await session.execute(
            sql_text(
                "SELECT COUNT(*) AS cnt FROM glyphs "
                "WHERE org_id = :org_id AND model_id = :model_id"
            ),
            {"org_id": org_id, "model_id": model_id},
        )
        total_count = total.scalar() or 0

        # Extension breakdown
        ext_result = await session.execute(
            sql_text(
                "SELECT metadata->>'extension' AS ext, COUNT(*) AS cnt "
                "FROM glyphs "
                "WHERE org_id = :org_id AND model_id = :model_id "
                "GROUP BY metadata->>'extension' "
                "ORDER BY cnt DESC"
            ),
            {"org_id": org_id, "model_id": model_id},
        )
        extensions = {r.ext: r.cnt for r in ext_result.fetchall() if r.ext}

    return _mcp_json({
        "state": "DONE",
        "total_files": total_count,
        "extensions": extensions,
        "model_id": model_id,
    })
