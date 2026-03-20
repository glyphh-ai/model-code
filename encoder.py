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

    Hierarchical vector storage: per-layer cortex vectors stored in
    glyph_vectors table. Search queries the appropriate layer based on
    structural query analysis, then re-ranks with layer-weighted scoring.

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
    apply_weights_during_encoding=True,
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

# Opt into hierarchical vector storage so per-layer cortex vectors
# are stored in the glyph_vectors table. This is read by the runtime
# listener from encoder_config._similarity_config in the DB.
ENCODER_CONFIG_EXTRA = {
    "_similarity_config": {
        "store_hierarchical_vectors": True,
    },
}


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

# Layer weights for scoring — must match ENCODER_CONFIG
_PATH_WEIGHT = 0.30
_CONTENT_WEIGHT = 0.70


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
        path_tokens = clean
        identifiers = clean
        imports = clean
    else:
        # Concept queries: populate ALL roles so both layers participate.
        # The layer weights (0.30 path + 0.70 content) handle the balance.
        # Without path_tokens, the path layer produces a random vector and
        # dual-layer coarse retrieval can't find files by path terms.
        path_tokens = clean
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

    return {
        "concept_text": rel_path,
        "attributes": {
            "path_tokens": path_tokens,
            "identifiers": identifiers,
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


# ---------------------------------------------------------------------------
# Layer-level pgvector search
# ---------------------------------------------------------------------------

async def _layer_search(
    session_factory: Any,
    org_id: str,
    model_id: str,
    query_vector,
    layer_path: str,
    top_k: int = 5,
    exclude_glyph_id: str | None = None,
) -> list[dict]:
    """Search glyph_vectors table for a specific layer.

    Queries the hierarchical vector storage by layer path (e.g., "content",
    "path") and returns matching glyph IDs with scores. Joins back to the
    glyphs table for concept_text and metadata.
    """
    import numpy as np
    from sqlalchemy import text

    vec = np.asarray(query_vector, dtype=float)
    vec_str = "[" + ",".join(f"{v:.1f}" for v in vec) + "]"

    exclude_clause = ""
    params: dict[str, Any] = {
        "org_id": org_id,
        "model_id": model_id,
        "vec": vec_str,
        "layer_path": layer_path,
        "top_k": top_k,
    }

    if exclude_glyph_id:
        exclude_clause = "AND gv.glyph_id != CAST(:exclude_glyph_id AS uuid)"
        params["exclude_glyph_id"] = exclude_glyph_id

    async with session_factory() as session:
        result = await session.execute(
            text(
                f"SELECT g.id AS glyph_id, g.concept_text, g.metadata, "
                f"1 - (gv.embedding <=> CAST(:vec AS vector)) AS score "
                f"FROM glyph_vectors gv "
                f"JOIN glyphs g ON gv.glyph_id = g.id "
                f"WHERE gv.org_id = :org_id AND gv.model_id = :model_id "
                f"AND gv.level = 'layer' AND gv.path = :layer_path "
                f"{exclude_clause} "
                f"ORDER BY gv.embedding <=> CAST(:vec AS vector) "
                f"LIMIT :top_k"
            ),
            params,
        )
        rows = result.fetchall()

    results = []
    for row in rows:
        meta = row.metadata if isinstance(row.metadata, dict) else {}
        results.append({
            "glyph_id": str(row.glyph_id),
            "concept_text": row.concept_text,
            "metadata": meta,
            "score": float(row.score),
        })
    return results


async def _rerank_with_layers(
    session_factory: Any,
    org_id: str,
    model_id: str,
    query_glyph,
    candidate_glyph_ids: list[str],
) -> list[dict]:
    """Re-rank candidates using layer-weighted scoring.

    For each candidate, loads its per-layer vectors from glyph_vectors
    and computes weighted similarity: 0.30 * path_sim + 0.70 * content_sim.
    """
    import numpy as np
    from glyphh.core.ops import cosine_similarity
    from sqlalchemy import text

    if not candidate_glyph_ids:
        return []

    # Get query layer cortexes
    query_path_cortex = None
    query_content_cortex = None
    if hasattr(query_glyph, "layers"):
        if "path" in query_glyph.layers:
            query_path_cortex = query_glyph.layers["path"].cortex
        if "content" in query_glyph.layers:
            query_content_cortex = query_glyph.layers["content"].cortex

    # Load candidate layer vectors
    id_list = ",".join(f"'{gid}'" for gid in candidate_glyph_ids)
    async with session_factory() as session:
        result = await session.execute(
            text(
                f"SELECT gv.glyph_id, gv.path, gv.embedding, "
                f"g.concept_text, g.metadata "
                f"FROM glyph_vectors gv "
                f"JOIN glyphs g ON gv.glyph_id = g.id "
                f"WHERE gv.org_id = :org_id AND gv.model_id = :model_id "
                f"AND gv.level = 'layer' "
                f"AND gv.glyph_id IN ({id_list})"
            ),
            {"org_id": org_id, "model_id": model_id},
        )
        rows = result.fetchall()

    # Group by glyph_id
    candidates: dict[str, dict] = {}
    for row in rows:
        gid = str(row.glyph_id)
        if gid not in candidates:
            meta = row.metadata if isinstance(row.metadata, dict) else {}
            candidates[gid] = {
                "concept_text": row.concept_text,
                "metadata": meta,
                "layers": {},
            }
        # Convert pgvector type to numpy array for cosine_similarity.
        # pgvector returns various types depending on driver: numpy array,
        # list of floats, or a pgvector Vector object with list-like access.
        embedding = row.embedding
        if isinstance(embedding, np.ndarray):
            arr = embedding.astype(np.int8)
        elif isinstance(embedding, (list, tuple)):
            arr = np.array(embedding, dtype=np.int8)
        elif hasattr(embedding, "to_list"):
            arr = np.array(embedding.to_list(), dtype=np.int8)
        else:
            # pgvector string: "[1,-1,1,...]" — parse manually
            arr = np.array(json.loads(str(embedding)), dtype=np.int8)
        candidates[gid]["layers"][row.path] = arr

    # Score each candidate with layer-weighted similarity
    scored = []
    for gid, cand in candidates.items():
        path_sim = 0.0
        content_sim = 0.0

        if query_path_cortex is not None and "path" in cand["layers"]:
            path_sim = float(cosine_similarity(
                query_path_cortex.data, cand["layers"]["path"]
            ))

        if query_content_cortex is not None and "content" in cand["layers"]:
            content_sim = float(cosine_similarity(
                query_content_cortex.data, cand["layers"]["content"]
            ))

        score = _PATH_WEIGHT * path_sim + _CONTENT_WEIGHT * content_sim

        scored.append({
            "glyph_id": gid,
            "concept_text": cand["concept_text"],
            "metadata": cand["metadata"],
            "score": score,
            "path_sim": path_sim,
            "content_sim": content_sim,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_match(row: dict) -> dict:
    """Format a search result into a match dict.

    The runtime stores the full record as glyph metadata. When records
    are sent in hierarchical format, the model's metadata dict is nested
    under the 'metadata' key. Handle both flat and nested layouts.
    """
    meta = row.get("metadata", {})
    inner = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
    return {
        "file": inner.get("file_path") or meta.get("file_path") or row["concept_text"],
        "confidence": round(row["score"], 3),
        "top_tokens": inner.get("top_tokens") or meta.get("top_tokens", []),
        "imports": inner.get("imports") or meta.get("imports", []),
        "extension": inner.get("extension") or meta.get("extension", ""),
    }


# ---------------------------------------------------------------------------
# Search handler — layer-aware
# ---------------------------------------------------------------------------

async def _handle_search(arguments: dict, context: dict) -> dict:
    """Find files by NL query using layer-level similarity.

    Strategy:
    1. Analyze query intent (structural, import, concept)
    2. Search the primary layer via glyph_vectors (content for concept
       queries, path for structural queries)
    3. Re-rank top candidates with full layer-weighted scoring
    4. Apply confidence gate (threshold + gap analysis)
    """
    from glyphh.core.types import Concept

    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    if not query.strip():
        return _mcp_text("Query is empty.")

    org_id = context["org_id"]
    model_id = context["model_id"]
    encoder = context["encoder"]
    encode_fn = context["encode_query_fn"]

    # Encode query into a full glyph with layer cortexes
    concept_dict = encode_fn(query)
    attrs = concept_dict.get("attributes", concept_dict)
    name = concept_dict.get("name", "query")
    concept = Concept(name=name, attributes=attrs)
    query_glyph = encoder.encode(concept)

    # Determine query intent for logging / future use
    tokens = _tokenize(query)
    words = [w for w in tokens.split() if w not in _STOP_WORDS and len(w) > 1]
    intent = _analyze_query(query, words)

    # Phase 1: coarse retrieval from BOTH layers, merge candidates
    # Queries often mix path terms ("studio hub") and content terms ("model"),
    # so searching only one layer misses files that match the other.
    coarse_k = max(top_k * 4, 20)
    seen_glyph_ids: set[str] = set()
    coarse_results: list[dict] = []

    has_layers = hasattr(query_glyph, "layers")

    # Search content layer
    if has_layers and "content" in query_glyph.layers:
        content_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.layers["content"].cortex.data,
            "content", top_k=coarse_k,
        )
        for r in content_results:
            gid = r["glyph_id"]
            if gid not in seen_glyph_ids:
                seen_glyph_ids.add(gid)
                coarse_results.append(r)

    # Search path layer
    if has_layers and "path" in query_glyph.layers:
        path_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.layers["path"].cortex.data,
            "path", top_k=coarse_k,
        )
        for r in path_results:
            gid = r["glyph_id"]
            if gid not in seen_glyph_ids:
                seen_glyph_ids.add(gid)
                coarse_results.append(r)

    # Fallback: search global cortex if no layer results
    if not coarse_results and has_layers:
        coarse_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.global_cortex.data,
            "content", top_k=coarse_k,
        )

    if not coarse_results:
        return _mcp_json({
            "state": "ASK",
            "message": "No files indexed. Run: glyphh-code compile .",
        })

    # Phase 2: re-rank with layer-weighted scoring
    candidate_ids = [r["glyph_id"] for r in coarse_results]
    ranked = await _rerank_with_layers(
        context["session_factory"],
        org_id,
        model_id,
        query_glyph,
        candidate_ids,
    )

    if not ranked:
        return _mcp_json({
            "state": "ASK",
            "message": "Encoding error during re-ranking.",
        })

    matches = [_format_match(r) for r in ranked[:top_k]]
    top_score = matches[0]["confidence"]
    gap = top_score - matches[1]["confidence"] if len(matches) > 1 else 1.0

    if top_score < 0.10 or gap < 0.02:
        return _mcp_json({
            "state": "ASK",
            "candidates": matches[:3],
            "message": "Multiple similar files found. Which did you mean?",
        })

    return _mcp_json({
        "state": "DONE",
        "matches": matches,
    })


# ---------------------------------------------------------------------------
# Related handler — layer-aware
# ---------------------------------------------------------------------------

async def _handle_related(arguments: dict, context: dict) -> dict:
    """Find files semantically related to a given file.

    Uses the content layer for similarity (files related by what they do,
    not where they are in the tree).
    """
    from sqlalchemy import text as sql_text

    file_path = arguments.get("file_path", "")
    top_k = arguments.get("top_k", 5)
    if not file_path.strip():
        return _mcp_text("file_path is required.")

    org_id = context["org_id"]
    model_id = context["model_id"]

    # Find the glyph by file path (concept_text or metadata)
    async with context["session_factory"]() as session:
        result = await session.execute(
            sql_text(
                "SELECT g.id, g.concept_text, g.metadata "
                "FROM glyphs g "
                "WHERE g.org_id = :org_id AND g.model_id = :model_id "
                "AND (g.concept_text = :file_path "
                "     OR g.metadata->'metadata'->>'file_path' = :file_path) "
                "LIMIT 1"
            ),
            {"org_id": org_id, "model_id": model_id, "file_path": file_path},
        )
        glyph_row = result.fetchone()

    if glyph_row is None:
        return _mcp_json({
            "state": "ASK",
            "message": f"File not in index: {file_path}. Run: glyphh-code compile .",
        })

    glyph_id = str(glyph_row.id)

    # Get the content layer vector for this file
    async with context["session_factory"]() as session:
        result = await session.execute(
            sql_text(
                "SELECT embedding FROM glyph_vectors "
                "WHERE glyph_id = CAST(:glyph_id AS uuid) "
                "AND level = 'layer' AND path = 'content'"
            ),
            {"glyph_id": glyph_id},
        )
        content_row = result.fetchone()

    if content_row is None:
        return _mcp_json({
            "state": "ASK",
            "message": f"No content vector for: {file_path}. Re-compile with hierarchical storage.",
        })

    # Search for similar files by content layer
    results = await _layer_search(
        context["session_factory"],
        org_id,
        model_id,
        content_row.embedding,
        "content",
        top_k=top_k + 1,
        exclude_glyph_id=glyph_id,
    )

    related = [_format_match(r) for r in results[:top_k]]
    source_meta = glyph_row.metadata if isinstance(glyph_row.metadata, dict) else {}
    source_inner = source_meta.get("metadata", {}) if isinstance(source_meta.get("metadata"), dict) else {}

    return _mcp_json({
        "state": "DONE",
        "file": file_path,
        "top_tokens": source_inner.get("top_tokens") or source_meta.get("top_tokens", []),
        "imports": source_inner.get("imports") or source_meta.get("imports", []),
        "related": related,
    })


# ---------------------------------------------------------------------------
# Stats handler
# ---------------------------------------------------------------------------

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

        # Hierarchical vectors
        vectors = await session.execute(
            sql_text(
                "SELECT COUNT(*) AS cnt FROM glyph_vectors "
                "WHERE org_id = :org_id AND model_id = :model_id"
            ),
            {"org_id": org_id, "model_id": model_id},
        )
        vector_count = vectors.scalar() or 0

        # Extension breakdown
        ext_result = await session.execute(
            sql_text(
                "SELECT metadata->'metadata'->>'extension' AS ext, COUNT(*) AS cnt "
                "FROM glyphs "
                "WHERE org_id = :org_id AND model_id = :model_id "
                "GROUP BY metadata->'metadata'->>'extension' "
                "ORDER BY cnt DESC"
            ),
            {"org_id": org_id, "model_id": model_id},
        )
        extensions = {r.ext: r.cnt for r in ext_result.fetchall() if r.ext}

    return _mcp_json({
        "state": "DONE",
        "total_files": total_count,
        "hierarchical_vectors": vector_count,
        "extensions": extensions,
        "model_id": model_id,
    })
