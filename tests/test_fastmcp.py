"""Real-world query routing tests using fastmcp as a test corpus.

Compiles actual source files from the fastmcp project and verifies
that NL queries route to the correct files. This tests the full
pipeline: file_to_record → encode → weighted layer similarity.

Requires: fastmcp source at ~/development/a-test/fastmcp
Skip gracefully if not available.
"""

import os
import pytest
from pathlib import Path

from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity

from glyphh_code.encoder import ENCODER_CONFIG, encode_query, file_to_record

# Layer weights — must match ENCODER_CONFIG
_LAYER_WEIGHTS = {"path": 0.25, "symbols": 0.25, "content": 0.50}

FASTMCP_ROOT = os.path.expanduser("~/development/a-test/fastmcp")


def _weighted_similarity(g1, g2):
    """Weighted layer similarity (excludes temporal noise)."""
    total = 0.0
    for name, weight in _LAYER_WEIGHTS.items():
        if name in g1.layers and name in g2.layers:
            sim = cosine_similarity(g1.layers[name].cortex.data, g2.layers[name].cortex.data)
            total += weight * sim
    return total


# Files to index — representative cross-section of the codebase
_FILES = [
    "src/fastmcp/server/auth/auth.py",
    "src/fastmcp/server/auth/oauth_proxy/__init__.py",
    "src/fastmcp/server/auth/jwt_issuer.py",
    "src/fastmcp/client/auth/oauth.py",
    "src/fastmcp/server/auth/middleware.py",
    "src/fastmcp/server/auth/authorization.py",
    "src/fastmcp/server/auth/ssrf.py",
    "src/fastmcp/tools/function_tool.py",
    "src/fastmcp/tools/base.py",
    "src/fastmcp/resources/base.py",
    "src/fastmcp/resources/function_resource.py",
    "src/fastmcp/cli/__init__.py",
    "src/fastmcp/settings.py",
    "src/fastmcp/exceptions.py",
    "src/fastmcp/decorators.py",
]


@pytest.fixture(scope="module")
def fastmcp_index():
    """Compile fastmcp files into an in-memory glyph index."""
    if not Path(FASTMCP_ROOT).is_dir():
        pytest.skip("fastmcp not available at ~/development/a-test/fastmcp")

    encoder = Encoder(ENCODER_CONFIG)
    index = {}
    for f in _FILES:
        full_path = os.path.join(FASTMCP_ROOT, f)
        record = file_to_record(full_path, FASTMCP_ROOT)
        if record:
            concept = Concept(name=record["concept_text"], attributes=record["attributes"])
            glyph = encoder.encode(concept)
            index[f] = glyph
    return encoder, index


def _query_top_k(fastmcp_index, query: str, k: int = 3) -> list[tuple[str, float]]:
    """Run a query against the index, return top-k (file, score) pairs."""
    encoder, index = fastmcp_index
    qr = encode_query(query)
    qc = Concept(name=qr["name"], attributes=qr["attributes"])
    qg = encoder.encode(qc)
    scores = {f: _weighted_similarity(qg, g) for f, g in index.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


class TestFastMCPRouting:
    """Test that NL queries route to the correct fastmcp files."""

    def test_oauth_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "where is oauth stored")]
        acceptable = {
            "src/fastmcp/server/auth/oauth_proxy/__init__.py",
            "src/fastmcp/client/auth/oauth.py",
        }
        assert acceptable & set(top3), f"Expected oauth file in top 3, got {top3}"

    def test_jwt_token_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "jwt token generation")]
        assert "src/fastmcp/server/auth/jwt_issuer.py" in top3

    def test_auth_middleware_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "authentication middleware")]
        acceptable = {
            "src/fastmcp/server/auth/auth.py",
            "src/fastmcp/server/auth/middleware.py",
        }
        assert acceptable & set(top3), f"Expected auth file in top 3, got {top3}"

    def test_tool_definition_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "tool definition decorator")]
        acceptable = {
            "src/fastmcp/tools/function_tool.py",
            "src/fastmcp/tools/base.py",
            "src/fastmcp/decorators.py",
        }
        assert acceptable & set(top3), f"Expected tool file in top 3, got {top3}"

    def test_resource_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "resource handling")]
        acceptable = {
            "src/fastmcp/resources/base.py",
            "src/fastmcp/resources/function_resource.py",
        }
        assert acceptable & set(top3), f"Expected resource file in top 3, got {top3}"

    def test_cli_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "cli commands")]
        assert "src/fastmcp/cli/__init__.py" in top3

    def test_settings_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "application settings configuration")]
        assert "src/fastmcp/settings.py" in top3

    def test_exceptions_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "error handling exceptions")]
        assert "src/fastmcp/exceptions.py" in top3

    def test_ssrf_query(self, fastmcp_index):
        top3 = [f for f, _ in _query_top_k(fastmcp_index, "ssrf protection validation")]
        assert "src/fastmcp/server/auth/ssrf.py" in top3

    def test_top_score_above_noise(self, fastmcp_index):
        """Top match should be well above random noise for specific queries."""
        top = _query_top_k(fastmcp_index, "jwt token generation", k=1)
        assert top[0][1] > 0.05, f"Top score {top[0][1]} too close to noise floor"


class TestFastMCPCompilation:
    """Test that real files compile with meaningful content."""

    def test_files_have_identifiers(self, fastmcp_index):
        """Every compiled file should produce a glyph."""
        _, index = fastmcp_index
        assert len(index) >= 10, f"Only {len(index)} files compiled"

    def test_auth_file_has_rich_vocabulary(self, fastmcp_index):
        """auth.py should have hundreds of identifier tokens."""
        if not Path(FASTMCP_ROOT).is_dir():
            pytest.skip("fastmcp not available")
        record = file_to_record(
            os.path.join(FASTMCP_ROOT, "src/fastmcp/server/auth/auth.py"),
            FASTMCP_ROOT,
        )
        assert record is not None
        n_tokens = len(record["attributes"]["identifiers"].split())
        assert n_tokens > 100, f"auth.py only has {n_tokens} identifier tokens"

    def test_ast_extraction_produces_defines(self, fastmcp_index):
        """Files with classes/functions should have AST-extracted defines."""
        if not Path(FASTMCP_ROOT).is_dir():
            pytest.skip("fastmcp not available")
        record = file_to_record(
            os.path.join(FASTMCP_ROOT, "src/fastmcp/server/auth/jwt_issuer.py"),
            FASTMCP_ROOT,
        )
        assert record is not None
        defines = record["attributes"]["defines"]
        assert len(defines) > 0, "jwt_issuer.py should have AST-extracted defines"
        assert "jwt" in defines.lower(), f"Expected 'jwt' in defines: {defines[:100]}"
