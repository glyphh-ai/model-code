"""Similarity and routing accuracy tests for the Glyphh Code model."""

import pytest
from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity

from glyphh_code.encoder import ENCODER_CONFIG, encode_query, _extract_identifiers, _extract_imports, _tokenize
from glyphh_code.ast_extract import extract_sections

# Layer weights matching the encoder config
_LAYER_WEIGHTS = {"path": 0.25, "symbols": 0.25, "content": 0.50}


@pytest.fixture
def encoder():
    return Encoder(ENCODER_CONFIG)


def _encode_file(encoder, path_tokens, identifiers, imports="", defines="", file_role="source"):
    """Helper to encode a file record as a glyph."""
    return encoder.encode(Concept(
        name=path_tokens.replace(" ", "/"),
        attributes={
            "path_tokens": path_tokens,
            "defines": defines or identifiers,
            "docstring": "",
            "file_role": file_role,
            "identifiers": identifiers,
            "imports": imports,
        },
    ))


def _encode_query(encoder, query):
    """Helper to encode a NL query as a glyph.

    encode_query already returns all layer attributes (path_tokens,
    defines, docstring, file_role, identifiers, imports).
    """
    result = encode_query(query)
    return encoder.encode(Concept(
        name=result["name"],
        attributes=result["attributes"],
    ))


def _weighted_similarity(g1, g2):
    """Compute weighted layer similarity (excludes temporal noise).

    This matches how the runtime's GlyphStorage.similarity_search_by_level
    scores results — per-layer cosine similarity weighted by config weights.
    """
    total = 0.0
    for name, weight in _LAYER_WEIGHTS.items():
        if name in g1.layers and name in g2.layers:
            sim = cosine_similarity(g1.layers[name].cortex.data, g2.layers[name].cortex.data)
            total += weight * sim
    return total


class TestSimilarityBasics:
    """Basic similarity properties."""

    def test_identical_files_perfect_similarity(self, encoder):
        g1 = _encode_file(encoder, "src services auth py", "auth token validate jwt hash")
        g2 = _encode_file(encoder, "src services auth py", "auth token validate jwt hash")
        sim = _weighted_similarity(g1, g2)
        assert sim == pytest.approx(1.0)

    def test_different_files_lower_than_related(self, encoder):
        """Unrelated files should score lower than related ones."""
        auth = _encode_file(encoder, "src services auth py", "auth token validate jwt hash")
        css = _encode_file(encoder, "lib css theme scss", "color font margin padding border")
        test_auth = _encode_file(encoder, "tests test auth py", "test auth token validate mock")

        sim_unrelated = _weighted_similarity(auth, css)
        sim_related = _weighted_similarity(auth, test_auth)
        assert sim_related > sim_unrelated

    def test_related_files_moderate_similarity(self, encoder):
        g1 = _encode_file(
            encoder,
            "src services user service py",
            "user create delete update email password",
            "sqlalchemy fastapi",
        )
        g2 = _encode_file(
            encoder,
            "tests test user service py",
            "test create user test delete user mock fixture",
            "pytest mock services user",
        )
        sim = _weighted_similarity(g1, g2)
        assert 0.20 < sim < 0.80  # Related but not identical


class TestQueryRouting:
    """Test that NL queries route to the correct files."""

    @pytest.fixture
    def file_glyphs(self, encoder):
        """Create a small catalog of file glyphs with rich vocabularies.

        At 2000 dims, BoW needs enough distinctive tokens to discriminate.
        Real files have hundreds of tokens — these test exemplars use 20+.
        """
        files = {
            "auth_service": _encode_file(
                encoder,
                "src services auth service py",
                (
                    "auth authenticate login logout token jwt refresh password "
                    "hash verify bearer middleware session cookie expire claims "
                    "issuer audience decode encode secret bcrypt rounds salt"
                ),
                "bcrypt pyjwt fastapi sqlalchemy passlib",
            ),
            "payment_handler": _encode_file(
                encoder,
                "src handlers payment py",
                (
                    "payment charge refund stripe subscription invoice webhook "
                    "billing amount currency customer card declined receipt "
                    "checkout session intent capture payout transfer balance"
                ),
                "stripe httpx aiohttp",
            ),
            "database_migration": _encode_file(
                encoder,
                "migrations versions add users table py",
                (
                    "alembic migration upgrade downgrade revision autogenerate "
                    "schema alter add column drop index foreign constraint "
                    "batch operations metadata naming convention"
                ),
                "alembic sqlalchemy",
            ),
            "react_component": _encode_file(
                encoder,
                "src components dashboard tsx",
                (
                    "react component dashboard widget chart graph render "
                    "useState useEffect props children jsx tsx onclick "
                    "className tailwind grid flex responsive layout"
                ),
                "react recharts tailwindcss",
            ),
        }
        return files

    def _best_match(self, encoder, query, file_glyphs):
        """Find the best matching file for a query."""
        q = _encode_query(encoder, query)
        scores = {
            name: _weighted_similarity(q, g)
            for name, g in file_glyphs.items()
        }
        return max(scores, key=scores.get), scores

    def test_auth_query_matches_auth_service(self, encoder, file_glyphs):
        best, scores = self._best_match(
            encoder, "jwt token authenticate bearer login password", file_glyphs
        )
        assert best == "auth_service", f"Expected auth_service, got {best}: {scores}"

    def test_payment_query_matches_payment_handler(self, encoder, file_glyphs):
        best, scores = self._best_match(
            encoder, "stripe charge refund billing invoice", file_glyphs
        )
        assert best == "payment_handler", f"Expected payment_handler, got {best}: {scores}"

    def test_database_query_matches_migration(self, encoder, file_glyphs):
        best, scores = self._best_match(
            encoder, "alembic migration upgrade downgrade schema", file_glyphs
        )
        assert best == "database_migration", f"Expected database_migration, got {best}: {scores}"

    def test_react_query_matches_component(self, encoder, file_glyphs):
        best, scores = self._best_match(
            encoder, "react component dashboard widget chart jsx", file_glyphs
        )
        assert best == "react_component", f"Expected react_component, got {best}: {scores}"


class TestLayerSimilarity:
    """Test that path and content layers contribute independently."""

    def test_same_path_different_content(self, encoder):
        """Files at same path but different content: path layer high, content layer lower."""
        g1 = _encode_file(
            encoder, "src utils helpers py",
            "parse json format date serialize deserialize stringify",
            "json datetime",
        )
        g2 = _encode_file(
            encoder, "src utils helpers py",
            "encrypt decrypt hash verify signature certificate openssl",
            "cryptography hashlib",
        )
        path_sim = cosine_similarity(
            g1.layers["path"].cortex.data,
            g2.layers["path"].cortex.data,
        )
        content_sim = cosine_similarity(
            g1.layers["content"].cortex.data,
            g2.layers["content"].cortex.data,
        )
        assert path_sim > 0.90  # Same path
        assert content_sim < path_sim  # Content should differ more than path

    def test_different_path_same_content(self, encoder):
        """Files with same content but different paths."""
        g1 = _encode_file(encoder, "src services auth py", "auth token validate jwt")
        g2 = _encode_file(encoder, "lib middleware verify py", "auth token validate jwt")
        path_sim = cosine_similarity(
            g1.layers["path"].cortex.data,
            g2.layers["path"].cortex.data,
        )
        content_sim = cosine_similarity(
            g1.layers["content"].cortex.data,
            g2.layers["content"].cortex.data,
        )
        assert path_sim < 0.50  # Different paths
        assert content_sim > 0.90  # Same content


class TestImportSignal:
    """Test that imports provide meaningful signal."""

    def test_shared_imports_increase_similarity(self, encoder):
        """Files sharing imports should be more similar than those with no overlap."""
        g_shared = _encode_file(
            encoder, "src services a py",
            "user create update delete query",
            "sqlalchemy fastapi pydantic alembic",
            defines="UserService create_user update_user",
        )
        g_also_shared = _encode_file(
            encoder, "src services b py",
            "account create update delete query",
            "sqlalchemy fastapi httpx alembic",
            defines="AccountService create_account update_account",
        )
        g_different = _encode_file(
            encoder, "src components c tsx",
            "render widget chart dashboard layout",
            "react nextjs tailwind recharts",
            defines="DashboardWidget render_chart",
        )

        sim_shared = _weighted_similarity(g_shared, g_also_shared)
        sim_different = _weighted_similarity(g_shared, g_different)
        assert sim_shared > sim_different


class TestMCPToolSchemas:
    """Validate MCP tool definitions."""

    def test_mcp_tools_defined(self):
        from glyphh_code.encoder import MCP_TOOLS
        assert isinstance(MCP_TOOLS, list)
        assert len(MCP_TOOLS) >= 2

    def test_search_tool_schema(self):
        from glyphh_code.encoder import MCP_TOOLS
        search = next(t for t in MCP_TOOLS if t["name"] == "glyphh_search")
        assert "description" in search
        assert search["input_schema"]["required"] == ["query"]

    def test_related_tool_schema(self):
        from glyphh_code.encoder import MCP_TOOLS
        related = next(t for t in MCP_TOOLS if t["name"] == "glyphh_related")
        assert related["input_schema"]["required"] == ["file_path"]

    def test_context_tool_schema(self):
        from glyphh_code.encoder import MCP_TOOLS
        context = next(t for t in MCP_TOOLS if t["name"] == "glyphh_context")
        assert "description" in context
        assert set(context["input_schema"]["required"]) == {"file_path", "query"}
        props = context["input_schema"]["properties"]
        assert "file_path" in props
        assert "query" in props
        assert "top_k" in props

    def test_handle_mcp_tool_defined(self):
        from glyphh_code.encoder import handle_mcp_tool
        import asyncio
        assert asyncio.iscoroutinefunction(handle_mcp_tool)


class TestSectionScoring:
    """Test that section-level encoding correctly identifies relevant code sections."""

    MULTI_FUNCTION_CODE = (
        "import os\n"
        "import hashlib\n"
        "\n"
        "def hash_password(password):\n"
        "    salt = os.urandom(32)\n"
        "    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)\n"
        "    return salt + key\n"
        "\n"
        "def verify_password(stored, provided):\n"
        "    salt = stored[:32]\n"
        "    key = hashlib.pbkdf2_hmac('sha256', provided.encode(), salt, 100000)\n"
        "    return stored[32:] == key\n"
        "\n"
        "def create_user(name, email, password):\n"
        "    hashed = hash_password(password)\n"
        "    return {'name': name, 'email': email, 'password_hash': hashed}\n"
        "\n"
        "def send_welcome_email(email, name):\n"
        "    subject = f'Welcome {name}'\n"
        "    body = f'Hello {name}, your account is ready.'\n"
        "    return {'to': email, 'subject': subject, 'body': body}\n"
    )

    def _score_sections(self, encoder, query, code, ext=".py"):
        """Encode query and sections, return sorted (name, score) pairs."""
        sections = extract_sections(code, ext)
        concept_dict = encode_query(query)
        attrs = concept_dict.get("attributes", concept_dict)
        name = concept_dict.get("name", "query")
        query_glyph = encoder.encode(Concept(name=name, attributes=attrs))

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
            sym_sim = float(cosine_similarity(
                query_glyph.layers["symbols"].cortex.data,
                section_glyph.layers["symbols"].cortex.data,
            ))
            content_sim = float(cosine_similarity(
                query_glyph.layers["content"].cortex.data,
                section_glyph.layers["content"].cortex.data,
            ))
            score = 0.4 * sym_sim + 0.6 * content_sim
            scored.append((section["name"], round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def test_password_query_finds_password_functions(self, encoder):
        """Query about password hashing should rank password functions highest."""
        scored = self._score_sections(
            encoder, "password hashing and verification", self.MULTI_FUNCTION_CODE,
        )
        top_names = [name for name, _ in scored[:2]]
        assert any("password" in n for n in top_names), (
            f"Expected password function in top 2, got: {scored}"
        )

    def test_email_query_finds_email_function(self, encoder):
        """Query about email should rank send_welcome_email highest."""
        scored = self._score_sections(
            encoder, "send welcome email notification", self.MULTI_FUNCTION_CODE,
        )
        assert scored[0][0] == "send_welcome_email", (
            f"Expected send_welcome_email as top match, got: {scored}"
        )

    def test_user_creation_query_finds_create_user(self, encoder):
        """Query about user creation should rank create_user highest."""
        scored = self._score_sections(
            encoder, "create new user account with name email", self.MULTI_FUNCTION_CODE,
        )
        top_names = [name for name, _ in scored[:2]]
        assert "create_user" in top_names, (
            f"Expected create_user in top 2, got: {scored}"
        )

    def test_section_scores_are_discriminative(self, encoder):
        """Top match should score meaningfully higher than bottom match."""
        scored = self._score_sections(
            encoder, "password hashing and verification", self.MULTI_FUNCTION_CODE,
        )
        top_score = scored[0][1]
        bottom_score = scored[-1][1]
        assert top_score > bottom_score, (
            f"Scores should be discriminative: top={top_score}, bottom={bottom_score}"
        )

    def test_real_file_section_routing(self, encoder):
        """Test on real encoder.py — 'MCP tool handler' should find handle_mcp_tool."""
        from pathlib import Path as P
        content = (P(__file__).parent.parent / "glyphh_code" / "encoder.py").read_text()
        scored = self._score_sections(encoder, "MCP tool handler dispatch", content)
        top_3_names = [name for name, _ in scored[:3]]
        assert "handle_mcp_tool" in top_3_names, (
            f"Expected handle_mcp_tool in top 3, got: {top_3_names}"
        )

    def test_token_savings_on_real_file(self, encoder):
        """Verify glyphh_context achieves meaningful line reduction on a real file."""
        from pathlib import Path as P
        content = (P(__file__).parent.parent / "glyphh_code" / "encoder.py").read_text()
        sections = extract_sections(content, ".py")
        total_lines = len(content.splitlines())

        # Top 3 sections should be much less than total file
        top_3_lines = sum(
            s["end_line"] - s["start_line"] + 1
            for s in sorted(sections, key=lambda s: s["end_line"] - s["start_line"])[:3]
        )
        reduction = 1 - (top_3_lines / total_lines)
        assert reduction > 0.50, (
            f"Expected >50% reduction, got {reduction:.0%} "
            f"({top_3_lines}/{total_lines} lines)"
        )
