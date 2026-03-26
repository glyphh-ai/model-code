"""
Encoder for the Glyphh Code model — file-level codebase intelligence.

Exports:
    ENCODER_CONFIG     — Four-layer HDC encoder (path + symbols + content + relationships)
    encode_query       — NL text → Concept dict for similarity search
    entry_to_record    — Raw file record → build record for runtime listener
    file_to_record     — Source file path → record dict (compile-time)
    MCP_TOOLS          — Model-specific MCP tool schemas
    handle_mcp_tool    — Async handler for model-specific MCP calls

Architecture:
    Path layer (0.20):          file path segments as BoW
    Symbols layer (0.25):       AST-extracted defines + module docstring as BoW
    Content layer (0.35):       identifiers (1.0) + imports (0.8) as BoW
    Relationships layer (0.20): dependents + references + co-change as BoW
    Dimension: 2000 (pgvector HNSW compatible)

    The symbols layer encodes what a file DEFINES (class/function names
    from tree-sitter AST), not what it USES. This naturally separates
    source files from their tests — auth.py defines AuthMiddleware while
    test_auth.py defines test_check_auth.

    The relationships layer encodes extrinsic signals: who imports this
    file, who uses its symbols, and which files change together in git
    history. Built from in-memory joins over already-extracted data —
    no subprocess/grep needed, fully cross-platform.

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
            similarity_weight=0.20,
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
            name="symbols",
            similarity_weight=0.25,
            segments=[
                Segment(
                    name="definitions",
                    roles=[
                        Role(
                            name="defines",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="docstring",
                            similarity_weight=0.5,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="file_role",
                            similarity_weight=0.3,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="content",
            similarity_weight=0.35,
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
        Layer(
            name="relationships",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="network",
                    roles=[
                        Role(
                            name="dependents",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="references",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="co_changed",
                            similarity_weight=0.6,
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
    # NL query noise — common in questions but never code identifiers
    "through", "between", "across", "within", "during", "about",
    "into", "also", "work", "works", "working", "does", "could",
    "would", "should", "might", "need", "needs", "when", "then",
    "there", "here", "been", "being", "have", "will", "just",
    "other", "some", "each", "every", "these", "those", "than",
    "them", "they", "its", "any", "may", "way", "like", "used",
    "use", "using", "make", "made", "would", "after", "before",
    "propagation", "involved", "manages", "deal", "deals",
    # Code noise
    "code", "file", "files", "function", "class", "method",
    "def", "var", "let", "const", "return", "import", "from",
    "true", "false", "none", "null", "self", "cls",
    "new", "try", "catch", "throw", "async", "await",
    "public", "private", "protected", "static", "void",
    "int", "str", "bool", "float", "string", "type",
})

# Layer weights for scoring — must match ENCODER_CONFIG
_PATH_WEIGHT = 0.25
_SYMBOLS_WEIGHT = 0.25
_CONTENT_WEIGHT = 0.50


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

    Three-layer encoding:
      - path_tokens: target + keywords → matches file paths
      - defines + docstring: query terms → matches AST-extracted symbols
      - identifiers + imports: keywords → matches raw code tokens

    The symbols layer carries the most weight (0.50) because AST-extracted
    defines encode WHAT a file is (its public API), which is what developers
    search for. This naturally separates source from test files.
    """
    from glyphh.intent import IntentExtractor

    global _intent_extractor
    if "_intent_extractor" not in globals() or _intent_extractor is None:
        _intent_extractor = IntentExtractor(packs=["code"])

    tokens = _tokenize(query)
    words = [w for w in tokens.split() if w not in _STOP_WORDS and len(w) > 1]
    clean = " ".join(words)

    struct = _analyze_query(query, words)
    intent = _intent_extractor.extract(query)

    target = intent.get("target", "")
    keywords = intent.get("keywords", clean)

    # Infer file role from query — "test", "spec" → test; "config", "yaml" → config; etc.
    query_lower = query.lower()
    _ROLE_SIGNALS = {
        "test": ("test", "spec", "unittest", "pytest"),
        "config": ("config", "yaml", "yml", "toml", "json config", "settings"),
        "docs": ("docs", "readme", "documentation", "markdown"),
        "script": ("script", "bash", "shell"),
        "example": ("example", "demo", "sample"),
    }
    file_role = "source"  # default: user is looking for source code
    for role, signals in _ROLE_SIGNALS.items():
        if any(s in query_lower for s in signals):
            file_role = role
            break

    if struct["is_structural"]:
        path_tokens = clean
        defines = clean
        docstring = clean
        identifiers = clean
        imports = ""
    else:
        # Path layer: target-focused for file name matching
        if target and target != "none":
            path_tokens = f"{target} {keywords}"
        else:
            path_tokens = keywords or clean

        # Symbols layer: query terms match AST-extracted symbol names
        defines = keywords or clean
        docstring = clean

        # Content layer: code-level tokens for disambiguation
        identifiers = clean
        imports = clean

    # Relationships layer: query tokens match against relationship path
    # tokens. When the user queries "auth middleware", files whose
    # dependents/references include "auth" or "middleware" path tokens
    # get a boost.  Empty string is fine — BoW of "" = zero vector,
    # so the relationships layer simply doesn't contribute to queries
    # that don't mention file-like terms.
    relationship_tokens = clean

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08x}",
        "attributes": {
            "path_tokens": path_tokens,
            "defines": defines,
            "docstring": docstring,
            "file_role": file_role,
            "identifiers": identifiers,
            "imports": imports,
            "dependents": relationship_tokens,
            "references": relationship_tokens,
            "co_changed": relationship_tokens,
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
            "defines": entry.get("defines", ""),
            "docstring": entry.get("docstring", ""),
            "file_role": entry.get("file_role", "source"),
            "identifiers": entry.get("identifiers", ""),
            "imports": entry.get("imports", ""),
            "dependents": entry.get("dependents", ""),
            "references": entry.get("references", ""),
            "co_changed": entry.get("co_changed", ""),
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

def file_to_record(
    file_path: str,
    repo_root: str = ".",
) -> dict | None:
    """Convert a source file to a Glyphh build record.

    Returns None for binary files, oversized files, or skipped types.
    Returns dict with concept_text, attributes, and metadata.

    Uses tree-sitter AST extraction (with regex fallback) to populate
    the symbols layer — what the file DEFINES, not what it USES.
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

    # AST extraction — deterministic, language-agnostic via tree-sitter
    from glyphh_code.ast_extract import extract_file_symbols
    symbols = extract_file_symbols(rel_path, content)
    defines = symbols.get("defines", "")
    docstring = symbols.get("docstring", "")
    file_role = symbols.get("file_role", "")
    ast_imports = symbols.get("imports", "")

    # Merge AST imports with regex imports for broader coverage
    if ast_imports:
        imports = f"{imports} {ast_imports}"

    return {
        "concept_text": rel_path,
        "attributes": {
            "path_tokens": path_tokens,
            "defines": defines,
            "docstring": docstring,
            "file_role": file_role,
            "identifiers": identifiers,
            "imports": imports,
            # Relationship roles — populated by compile.py post-processing
            # via build_relationship_graph() after all files are extracted.
            "dependents": "",
            "references": "",
            "co_changed": "",
        },
        "metadata": {
            "file_path": rel_path,
            "extension": path.suffix,
            "file_size": path.stat().st_size,
            "top_tokens": top_tokens,
            "imports": [t for t in imports.split() if len(t) > 2],
            "file_role": file_role,
            "docstring": docstring[:100] if docstring else "",
        },
    }


# ---------------------------------------------------------------------------
# MCP Tools — exposed through the runtime's MCP server
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "glyphh_search",
        "description": (
            "Semantic codebase search — find files by concept, not string match. "
            "Use for queries Grep cannot answer: 'what handles webhook validation', "
            "'files related to the payment retry flow', 'authentication middleware chain'. "
            "Returns file paths with confidence scores. Use detail='minimal' for "
            "lightweight responses (file + score only)."
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
                "detail": {
                    "type": "string",
                    "enum": ["minimal", "full"],
                    "default": "minimal",
                    "description": (
                        "Response detail level. 'minimal' (default) returns "
                        "only file paths and confidence scores. 'full' includes "
                        "top_tokens, imports, and extension for each file."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "glyphh_related",
        "description": (
            "Find files semantically related to a given file — blast radius analysis. "
            "Call before editing to find files that may need coordinated changes. "
            "Returns files that share vocabulary, imports, or domain concepts. "
            "No Grep equivalent for this."
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
                "detail": {
                    "type": "string",
                    "enum": ["minimal", "full"],
                    "default": "minimal",
                    "description": (
                        "Response detail level. 'minimal' (default) returns "
                        "only file paths and confidence scores. 'full' includes "
                        "top_tokens, imports, and extension for each file."
                        "imports, and extension for each file. 'minimal' returns "
                        "only file paths and confidence scores."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "glyphh_context",
        "description": (
            "Read only the relevant sections of a file for a given query. "
            "Instead of reading the entire file, returns the top matching "
            "code sections (functions, classes) with line ranges. "
            "Use this INSTEAD of the Read tool when you have a specific "
            "question about a file. Reduces token usage by ~80%."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)",
                },
                "query": {
                    "type": "string",
                    "description": "What you are looking for in this file",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of sections to return",
                },
            },
            "required": ["file_path", "query"],
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
    {
        "name": "glyphh_drift",
        "description": (
            "Semantic drift score for a file — how much has its meaning changed "
            "since the last index build? Returns a drift score (0.0 = identical, "
            "1.0 = complete rewrite) and a label: cosmetic, moderate, significant, "
            "or architectural. No Grep equivalent — measures semantic change, not "
            "textual diff size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to score drift for",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "glyphh_risk",
        "description": (
            "Risk profile for changed files — aggregates semantic drift across "
            "all files modified since the last index build (or a specific git ref). "
            "Returns per-file drift scores, overall risk label, and hot files "
            "that may need human review. No Grep equivalent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "git_ref": {
                    "type": "string",
                    "default": "HEAD",
                    "description": (
                        "Git ref to compare against (default: HEAD). "
                        "Use 'HEAD~1' for the last commit, a branch name, or a commit hash."
                    ),
                },
            },
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
        "glyphh_context": _handle_context,
        "glyphh_stats": _handle_stats,
        "glyphh_drift": _handle_drift,
        "glyphh_risk": _handle_risk,
    }

    handler = handlers.get(tool_name)
    if handler is None:
        return {"state": "ERROR", "error": f"Unknown tool: {tool_name}"}

    try:
        return await handler(arguments, context)
    except Exception as e:
        return {"state": "ERROR", "error": f"Error in {tool_name}: {e}"}


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

    Uses the runtime's GlyphStorage.similarity_search_by_level() which
    handles both pgvector and SQLite backends transparently.
    """
    import numpy as np
    from domains.models.storage import GlyphStorage
    from uuid import UUID

    vec = np.asarray(query_vector, dtype=float).tolist()

    # Fetch extra results if we need to exclude one
    fetch_k = top_k + 1 if exclude_glyph_id else top_k

    async with session_factory() as session:
        storage = GlyphStorage(session)
        raw_results = await storage.similarity_search_by_level(
            org_id, model_id, vec, "layer", layer_path, fetch_k,
        )
        # raw_results: List[Tuple[UUID, str, float]] = (glyph_id, path, score)

        # Filter excluded glyph
        if exclude_glyph_id:
            raw_results = [
                r for r in raw_results if str(r[0]) != exclude_glyph_id
            ][:top_k]

        if not raw_results:
            return []

        # Batch-fetch glyph metadata
        glyph_ids = [r[0] for r in raw_results]
        glyphs_map = await storage.get_glyphs_by_ids(org_id, model_id, glyph_ids)

    results = []
    for glyph_id, _path, score in raw_results:
        gid = str(glyph_id)
        glyph = glyphs_map.get(gid)
        meta = glyph.metadata if glyph and glyph.metadata else {}
        concept_text = glyph.concept_text if glyph else ""
        results.append({
            "glyph_id": gid,
            "concept_text": concept_text,
            "metadata": meta,
            "score": float(score),
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

    For each candidate, loads its per-layer vectors via the runtime's
    GlyphStorage and computes weighted similarity:
      0.30 * path + 0.50 * symbols + 0.20 * content
    """
    import numpy as np
    from glyphh.core.ops import cosine_similarity
    from domains.models.storage import GlyphStorage
    from uuid import UUID

    if not candidate_glyph_ids:
        return []

    # Get query layer cortexes
    query_path_cortex = None
    query_symbols_cortex = None
    query_content_cortex = None
    if hasattr(query_glyph, "layers"):
        if "path" in query_glyph.layers:
            query_path_cortex = query_glyph.layers["path"].cortex
        if "symbols" in query_glyph.layers:
            query_symbols_cortex = query_glyph.layers["symbols"].cortex
        if "content" in query_glyph.layers:
            query_content_cortex = query_glyph.layers["content"].cortex

    # Load candidate layer vectors and metadata via runtime storage
    glyph_uuids = [UUID(gid) for gid in candidate_glyph_ids]
    async with session_factory() as session:
        storage = GlyphStorage(session)
        embeddings_map = await storage.get_hierarchical_embeddings(
            org_id, model_id, glyph_uuids,
        )
        glyphs_map = await storage.get_glyphs_by_ids(
            org_id, model_id, glyph_uuids,
        )

    # Build candidates dict from runtime data
    candidates: dict[str, dict] = {}
    for gid_str, levels in embeddings_map.items():
        glyph = glyphs_map.get(gid_str)
        meta = glyph.metadata if glyph and glyph.metadata else {}
        concept_text = glyph.concept_text if glyph else ""
        layers = {}
        for path_name, emb in (levels.get("layer") or {}).items():
            layers[path_name] = np.array(emb, dtype=np.int8)
        candidates[gid_str] = {
            "concept_text": concept_text,
            "metadata": meta,
            "layers": layers,
        }

    # Score each candidate with layer-weighted similarity
    scored = []
    for gid, cand in candidates.items():
        path_sim = 0.0
        symbols_sim = 0.0
        content_sim = 0.0

        if query_path_cortex is not None and "path" in cand["layers"]:
            path_sim = float(cosine_similarity(
                query_path_cortex.data, cand["layers"]["path"]
            ))

        if query_symbols_cortex is not None and "symbols" in cand["layers"]:
            symbols_sim = float(cosine_similarity(
                query_symbols_cortex.data, cand["layers"]["symbols"]
            ))

        if query_content_cortex is not None and "content" in cand["layers"]:
            content_sim = float(cosine_similarity(
                query_content_cortex.data, cand["layers"]["content"]
            ))

        score = (_PATH_WEIGHT * path_sim
                 + _SYMBOLS_WEIGHT * symbols_sim
                 + _CONTENT_WEIGHT * content_sim)

        scored.append({
            "glyph_id": gid,
            "concept_text": cand["concept_text"],
            "metadata": cand["metadata"],
            "score": score,
            "path_sim": path_sim,
            "symbols_sim": symbols_sim,
            "content_sim": content_sim,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_match(row: dict, detail: str = "full") -> dict:
    """Format a search result into a fact_tree child node.

    The runtime stores the full record as glyph metadata. When records
    are sent in hierarchical format, the model's metadata dict is nested
    under the 'metadata' key. Handle both flat and nested layouts.

    Args:
        detail: "full" includes top_tokens, imports, extension.
                "minimal" returns only file path and confidence score.

    Returns a FactNode-compatible dict: description, value, data_sample.
    """
    meta = row.get("metadata", {})
    inner = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
    file_path = inner.get("file_path") or meta.get("file_path") or row["concept_text"]
    score = round(row["score"], 3)

    if detail == "minimal":
        return {
            "description": file_path,
            "value": score,
            "children": [],
            "citations": [],
            "data_sample": {
                "file": file_path,
                "confidence": score,
            },
        }

    top_tokens = inner.get("top_tokens") or meta.get("top_tokens", [])
    imports = inner.get("imports") or meta.get("imports", [])
    extension = inner.get("extension") or meta.get("extension", "")

    return {
        "description": file_path,
        "value": score,
        "children": [],
        "citations": [],
        "data_sample": {
            "file": file_path,
            "confidence": score,
            "top_tokens": top_tokens,
            "imports": imports,
            "extension": extension,
        },
        "data_context": {},
    }


# ---------------------------------------------------------------------------
# Query decomposition — split complex NL into focused sub-queries
# ---------------------------------------------------------------------------

_DECOMPOSE_THRESHOLD = 3  # decompose when >3 meaningful tokens remain

def _decompose_query(query: str) -> list[str] | None:
    """Split a long NL query into overlapping pair sub-queries.

    Returns None if query is short enough to search directly or
    contains a file path (which should be searched intact).
    Otherwise returns a list of 2-token sub-queries built from
    overlapping pairs of the filtered token list.

    Example: "error propagation from tools through middleware to client"
      → filtered: ["error", "tools", "middleware", "client"]
      → pairs: ["error tools", "error middleware", "tools middleware",
                 "middleware client"]
    """
    # Skip decomposition for queries containing file paths
    if "/" in query and ".py" in query:
        return None
    tokens = _tokenize(query)
    words = [w for w in tokens.split() if w not in _STOP_WORDS and len(w) > 1]
    if len(words) <= _DECOMPOSE_THRESHOLD:
        return None
    # Generate overlapping pairs — each token paired with its neighbors
    # plus skip-one connections for coverage
    pairs: list[str] = []
    seen: set[str] = set()
    for i in range(len(words)):
        for j in range(i + 1, min(i + 3, len(words))):
            pair = f"{words[i]} {words[j]}"
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


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
    detail = arguments.get("detail", "minimal")
    if not query.strip():
        return {"state": "ERROR", "error": "Query is empty."}

    # Query decomposition: if query has many NL tokens, split into focused
    # sub-queries, run each, and merge results by max confidence.
    sub_queries = _decompose_query(query)
    if sub_queries is not None:
        merged: dict[str, dict] = {}  # file_path → best match dict
        for sq in sub_queries:
            sub_result = await _handle_search(
                {**arguments, "query": sq, "top_k": top_k},
                context,
            )
            if sub_result.get("state") != "DONE":
                continue
            ft = sub_result.get("fact_tree")
            if not ft:
                continue
            for child in ft.get("children", []):
                fp = child.get("description", "")
                score = child.get("value", 0.0)
                if fp not in merged or score > merged[fp]["value"]:
                    merged[fp] = child
        if merged:
            ranked = sorted(merged.values(), key=lambda c: c["value"], reverse=True)
            children = ranked[:top_k]
            result = {
                "state": "DONE",
                "fact_tree": {
                    "description": "Similarity Computation",
                    "value": None,
                    "children": children,
                    "citations": [],
                    "data_context": {"match_phase": "decomposed"},
                },
                "confidence": children[0]["value"],
                "match_method": "code_search_decomposed",
            }
            if detail == "minimal":
                result["_detail"] = "minimal"
            return result
        # Decomposition found nothing — fall through to direct search

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

    # Phase 1: path-only search — high precision for navigational queries.
    # If path layer produces a clear winner, skip the expensive combined search.
    coarse_k = max(top_k * 4, 20)
    has_layers = hasattr(query_glyph, "layers")
    path_results: list[dict] = []

    if has_layers and "path" in query_glyph.layers:
        path_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.layers["path"].cortex.data,
            "path", top_k=coarse_k,
        )

    # Check for strong path match — clear gap between #1 and #2
    PATH_CLEAR_THRESHOLD = 0.03  # 3% gap = confident path match
    if len(path_results) >= 2:
        path_gap = path_results[0]["score"] - path_results[1]["score"]
        if path_gap >= PATH_CLEAR_THRESHOLD and path_results[0]["score"] > 0.10:
            # Strong path match — re-rank just the path candidates with full scoring
            candidate_ids = [r["glyph_id"] for r in path_results[:top_k * 2]]
            ranked = await _rerank_with_layers(
                context["session_factory"], org_id, model_id,
                query_glyph, candidate_ids,
            )
            if ranked:
                children = [_format_match(r, detail) for r in ranked[:top_k]]
                result = {
                    "state": "DONE",
                    "fact_tree": {
                        "description": "Similarity Computation",
                        "value": None,
                        "children": children,
                        "citations": [],
                        "data_context": {"match_phase": "path_shortcut"},
                    },
                    "confidence": children[0]["value"],
                    "match_method": "code_search_path",
                }
                if detail == "minimal":
                    result["_detail"] = "minimal"
                return result

    # Phase 2: full search — merge candidates from all layers.
    # Cap each layer so none dominates the candidate pool.
    layer_cap = max(top_k * 2, 10)
    seen_glyph_ids: set[str] = set()
    layer_hits: dict[str, set[str]] = {"path": set(), "symbols": set(), "content": set()}
    coarse_results: list[dict] = []

    # Add path results first (already retrieved), capped
    for r in path_results[:layer_cap]:
        gid = r["glyph_id"]
        if gid not in seen_glyph_ids:
            seen_glyph_ids.add(gid)
            coarse_results.append(r)
        layer_hits["path"].add(gid)

    # Search symbols layer (primary search signal — AST defines)
    if has_layers and "symbols" in query_glyph.layers:
        symbols_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.layers["symbols"].cortex.data,
            "symbols", top_k=layer_cap,
        )
        for r in symbols_results:
            gid = r["glyph_id"]
            layer_hits["symbols"].add(gid)
            if gid not in seen_glyph_ids:
                seen_glyph_ids.add(gid)
                coarse_results.append(r)

    # Search content layer
    if has_layers and "content" in query_glyph.layers:
        content_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.layers["content"].cortex.data,
            "content", top_k=layer_cap,
        )
        for r in content_results:
            gid = r["glyph_id"]
            layer_hits["content"].add(gid)
            if gid not in seen_glyph_ids:
                seen_glyph_ids.add(gid)
                coarse_results.append(r)

    # Files found by multiple layers are stronger candidates — place at front.
    def _layer_count(gid: str) -> int:
        return sum(1 for s in layer_hits.values() if gid in s)

    multi = [r for r in coarse_results if _layer_count(r["glyph_id"]) > 1]
    single = [r for r in coarse_results if _layer_count(r["glyph_id"]) == 1]
    coarse_results = multi + single

    # Fallback: search global cortex if no layer results
    if not coarse_results and has_layers:
        coarse_results = await _layer_search(
            context["session_factory"], org_id, model_id,
            query_glyph.global_cortex.data,
            "content", top_k=coarse_k,
        )

    if not coarse_results:
        return {
            "state": "ASK",
            "fact_tree": None,
            "confidence": 0.0,
            "match_method": "none",
            "error": "No files indexed. Run: glyphh-code compile .",
        }

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
        return {
            "state": "ASK",
            "fact_tree": None,
            "confidence": 0.0,
            "match_method": "none",
            "error": "Encoding error during re-ranking.",
        }

    children = [_format_match(r, detail) for r in ranked[:top_k]]
    top_score = children[0]["value"]

    result = {
        "state": "DONE",
        "fact_tree": {
            "description": "Similarity Computation",
            "value": None,
            "children": children,
            "citations": [],
            "data_context": {},
        },
        "confidence": top_score,
        "match_method": "code_search",
    }
    if detail == "minimal":
        result["_detail"] = "minimal"
    return result


# ---------------------------------------------------------------------------
# Related handler — layer-aware
# ---------------------------------------------------------------------------

async def _handle_related(arguments: dict, context: dict) -> dict:
    """Find files semantically related to a given file.

    Uses the content layer for similarity (files related by what they do,
    not where they are in the tree).
    """
    import numpy as np
    from domains.models.storage import GlyphStorage
    from domains.models.db_models import Glyph
    from sqlalchemy import select

    file_path = arguments.get("file_path", "")
    top_k = arguments.get("top_k", 5)
    detail = arguments.get("detail", "minimal")
    if not file_path.strip():
        return {"state": "ERROR", "error": "file_path is required."}

    org_id = context["org_id"]
    model_id = context["model_id"]

    # Find the glyph by file path (concept_text or metadata) using ORM
    async with context["session_factory"]() as session:
        result = await session.execute(
            select(Glyph).where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
                Glyph.concept_text == file_path,
            ).limit(1)
        )
        glyph_row = result.scalars().first()

    if glyph_row is None:
        return {
            "state": "ASK",
            "fact_tree": None,
            "confidence": 0.0,
            "match_method": "none",
            "error": f"File not in index: {file_path}. Run: glyphh-code compile .",
        }

    glyph_id = str(glyph_row.id)
    glyph_meta = glyph_row.glyph_metadata if glyph_row.glyph_metadata else {}
    glyph_concept = glyph_row.concept_text

    # Get the content layer vector via runtime storage
    async with context["session_factory"]() as session:
        storage = GlyphStorage(session)
        embeddings_map = await storage.get_hierarchical_embeddings(
            org_id, model_id, [glyph_row.id],
        )

    glyph_embs = embeddings_map.get(glyph_id, {}).get("layer", {})
    content_emb = glyph_embs.get("content")

    if content_emb is None:
        return {
            "state": "ASK",
            "fact_tree": None,
            "confidence": 0.0,
            "match_method": "none",
            "error": f"No content vector for: {file_path}. Re-compile with hierarchical storage.",
        }

    content_vec = np.array(content_emb, dtype=np.int8)

    # Search for similar files by content layer
    results = await _layer_search(
        context["session_factory"],
        org_id,
        model_id,
        content_vec,
        "content",
        top_k=top_k + 1,
        exclude_glyph_id=glyph_id,
    )

    children = [_format_match(r, detail) for r in results[:top_k]]
    source_meta = glyph_meta if isinstance(glyph_meta, dict) else {}
    source_inner = source_meta.get("metadata", {}) if isinstance(source_meta.get("metadata"), dict) else {}

    data_context = {"source_file": file_path}
    if detail != "minimal":
        data_context["top_tokens"] = source_inner.get("top_tokens") or source_meta.get("top_tokens", [])
        data_context["imports"] = source_inner.get("imports") or source_meta.get("imports", [])

    result = {
        "state": "DONE",
        "fact_tree": {
            "description": "Related Files",
            "value": None,
            "children": children,
            "citations": [],
            "data_context": data_context,
        },
        "confidence": children[0]["value"] if children else 0.0,
        "match_method": "code_related",
    }
    if detail == "minimal":
        result["_detail"] = "minimal"
    return result


# ---------------------------------------------------------------------------
# Context handler — section-level file reading
# ---------------------------------------------------------------------------

async def _handle_context(arguments: dict, context: dict) -> dict:
    """Return only the relevant sections of a file for a given query.

    Pipeline:
    1. Read the file and split into sections via tree-sitter AST
    2. Encode each section as a mini-concept (symbols + content layers)
    3. Encode the query using the standard encode_query pipeline
    4. Score each section against the query using layer cosine similarity
    5. Return top-k sections with source code and line ranges

    This replaces full file reads for targeted queries, cutting token
    usage by ~80% (e.g., 60 lines returned instead of 500).
    """
    from glyphh.core.types import Concept
    from glyphh.core.ops import cosine_similarity
    from glyphh_code.ast_extract import extract_sections

    file_path = arguments.get("file_path", "")
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 3)

    if not file_path.strip():
        return {"state": "ERROR", "error": "file_path is required."}
    if not query.strip():
        return {"state": "ERROR", "error": "query is required."}

    # Resolve file path — try as-is, then relative to CWD
    path = Path(file_path)
    if not path.exists():
        path = Path.cwd() / file_path
    if not path.exists():
        return {"state": "ERROR", "error": f"File not found: {file_path}"}
    if path.stat().st_size > MAX_FILE_BYTES:
        return {"state": "ERROR", "error": f"File too large: {path.stat().st_size} bytes"}

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"state": "ERROR", "error": f"Cannot read file: {e}"}

    ext = path.suffix
    lines = content.split("\n")
    total_lines = len(lines)

    # Extract sections with line ranges
    sections = extract_sections(content, ext)
    if not sections:
        return {
            "state": "DONE",
            "file": file_path,
            "sections": [{
                "name": "__module__",
                "start_line": 1,
                "end_line": total_lines,
                "similarity": 1.0,
                "content": content,
            }],
            "total_lines": total_lines,
            "lines_returned": total_lines,
        }

    # Encode query
    encoder = context["encoder"]
    encode_fn = context["encode_query_fn"]
    concept_dict = encode_fn(query)
    attrs = concept_dict.get("attributes", concept_dict)
    name = concept_dict.get("name", "query")
    query_glyph = encoder.encode(Concept(name=name, attributes=attrs))

    # Encode each section and score against query
    scored = []
    for section in sections:
        section_identifiers = _extract_identifiers(section["content"])
        section_imports = _extract_imports(section["content"])

        section_attrs = {
            "path_tokens": "",
            "defines": _tokenize(section["name"]) if section["name"] != "__preamble__" else "",
            "docstring": "",
            "file_role": "source",
            "identifiers": section_identifiers,
            "imports": section_imports,
        }
        section_glyph = encoder.encode(Concept(
            name=f"section_{section['name']}",
            attributes=section_attrs,
        ))

        # Score using symbols + content layers (path is irrelevant within a file)
        sym_sim = 0.0
        content_sim = 0.0
        has_layers = hasattr(query_glyph, "layers")

        if has_layers and "symbols" in query_glyph.layers and "symbols" in section_glyph.layers:
            sym_sim = float(cosine_similarity(
                query_glyph.layers["symbols"].cortex.data,
                section_glyph.layers["symbols"].cortex.data,
            ))
        if has_layers and "content" in query_glyph.layers and "content" in section_glyph.layers:
            content_sim = float(cosine_similarity(
                query_glyph.layers["content"].cortex.data,
                section_glyph.layers["content"].cortex.data,
            ))

        # Weight: 40% symbol match + 60% content match
        score = 0.4 * sym_sim + 0.6 * content_sim
        scored.append({**section, "similarity": round(score, 3)})

    # Sort by score descending, return top-k
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_sections = scored[:top_k]
    lines_returned = sum(s["end_line"] - s["start_line"] + 1 for s in top_sections)

    return {
        "state": "DONE",
        "file": file_path,
        "sections": [
            {
                "name": s["name"],
                "start_line": s["start_line"],
                "end_line": s["end_line"],
                "similarity": s["similarity"],
                "content": s["content"],
            }
            for s in top_sections
        ],
        "total_lines": total_lines,
        "lines_returned": lines_returned,
    }


# ---------------------------------------------------------------------------
# Stats handler
# ---------------------------------------------------------------------------

async def _handle_stats(arguments: dict, context: dict) -> dict:
    """Return index statistics."""
    from domains.models.storage import GlyphStorage

    org_id = context["org_id"]
    model_id = context["model_id"]

    async with context["session_factory"]() as session:
        storage = GlyphStorage(session)
        total_count = await storage.count_glyphs(org_id, model_id)
        vector_count = await storage.count_glyph_vectors(org_id, model_id)

        # Extension breakdown — fetch all glyphs and count by extension
        glyphs = await storage.list_glyphs(org_id, model_id, limit=100000)
        ext_counter: dict[str, int] = {}
        for g in glyphs:
            meta = g.metadata if isinstance(g.metadata, dict) else {}
            inner = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
            ext = inner.get("extension") or meta.get("extension")
            if ext:
                ext_counter[ext] = ext_counter.get(ext, 0) + 1
        # Sort by count descending
        extensions = dict(sorted(ext_counter.items(), key=lambda x: x[1], reverse=True))

    return {
        "state": "DONE",
        "fact_tree": {
            "description": "Index Statistics",
            "value": {"total_files": total_count, "hierarchical_vectors": vector_count},
            "children": [],
            "citations": [],
            "data_context": {"extensions": extensions, "model_id": model_id},
        },
        "confidence": 1.0,
        "match_method": "stats",
    }


# ---------------------------------------------------------------------------
# Drift handler — semantic change since last index
# ---------------------------------------------------------------------------

async def _handle_drift(arguments: dict, context: dict) -> dict:
    """Compute semantic drift for a single file.

    Compares the stored glyph (from last compile) with a freshly-encoded
    version of the file on disk. Returns drift score and label.
    """
    import numpy as np
    from glyphh.core.types import Concept
    from domains.models.db_models import Glyph
    from sqlalchemy import select

    file_path = arguments.get("file_path", "")
    if not file_path.strip():
        return {"state": "ERROR", "error": "file_path is required."}

    org_id = context["org_id"]
    model_id = context["model_id"]
    encoder = context["encoder"]

    # 1. Look up stored glyph
    async with context["session_factory"]() as session:
        result = await session.execute(
            select(Glyph).where(
                Glyph.org_id == org_id,
                Glyph.model_id == model_id,
                Glyph.concept_text == file_path,
            ).limit(1)
        )
        glyph_row = result.scalars().first()

    if glyph_row is None:
        return {
            "state": "ASK",
            "error": f"File not in index: {file_path}. Run: glyphh-code compile .",
        }

    stored_vec = np.array(glyph_row.embedding, dtype=np.int8)

    # 2. Re-encode current file from disk
    # CWD of the server should be the repo root (uvicorn runs from there)
    # file_path is relative (concept_text = relative path from compile)
    abs_path = Path(file_path)
    if not abs_path.is_absolute():
        abs_path = Path(".") / file_path

    record = file_to_record(str(abs_path), ".")
    if record is None:
        return {
            "state": "ASK",
            "error": f"Cannot read or encode file: {file_path}",
        }

    concept = Concept(
        name=record["concept_text"],
        attributes=record["attributes"],
    )
    current_glyph = encoder.encode(concept)
    current_vec = np.array(current_glyph.global_cortex.data, dtype=np.int8)

    # 3. Compute drift
    from glyphh_code.drift import compute_drift, drift_label

    score = compute_drift(stored_vec, current_vec)
    label = drift_label(score)

    return {
        "state": "DONE",
        "file": file_path,
        "drift_score": score,
        "drift_label": label,
    }


# ---------------------------------------------------------------------------
# Risk handler — aggregate drift for changed files
# ---------------------------------------------------------------------------

async def _handle_risk(arguments: dict, context: dict) -> dict:
    """Score commit risk by aggregating per-file semantic drift.

    Gets changed files from git, computes drift for each, and returns
    an aggregate risk profile.
    """
    import subprocess

    git_ref = arguments.get("git_ref", "HEAD")
    repo_root = context.get("repo_root", ".")

    # Get list of changed files from git
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", git_ref],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        if result.returncode != 0:
            # Try as a commit — diff against parent
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{git_ref}~1", git_ref],
                capture_output=True, text=True, cwd=repo_root, timeout=10,
            )
        changed_files = [
            f.strip() for f in result.stdout.strip().split("\n")
            if f.strip() and Path(f.strip()).suffix in INDEXABLE_EXTENSIONS
        ]
    except Exception as e:
        return {"state": "ERROR", "error": f"git error: {e}"}

    if not changed_files:
        return {
            "state": "DONE",
            "risk_label": "cosmetic",
            "files": {},
            "max_drift": 0.0,
            "mean_drift": 0.0,
            "hot_files": [],
            "message": "No indexable files changed.",
        }

    # Compute drift for each changed file
    drift_scores: dict[str, float] = {}
    errors: list[str] = []
    for fp in changed_files:
        r = await _handle_drift(
            {"file_path": fp},
            context,
        )
        if r.get("state") == "DONE":
            drift_scores[fp] = r["drift_score"]
        else:
            errors.append(f"{fp}: {r.get('error', 'unknown')}")

    from glyphh_code.drift import score_commit_files

    risk = score_commit_files(drift_scores)
    risk["state"] = "DONE"
    if errors:
        risk["errors"] = errors
    return risk
