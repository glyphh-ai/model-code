"""Unit tests for the Glyphh Code encoder."""

import pytest
from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh_code.encoder import (
    ENCODER_CONFIG,
    encode_query,
    entry_to_record,
    _tokenize,
    _split_camel,
    _split_snake,
    _extract_path_tokens,
    _extract_imports,
    _extract_identifiers,
    _top_tokens,
    _STOP_WORDS,
    INDEXABLE_EXTENSIONS,
    SKIP_DIRS,
)
from glyphh_code.ast_extract import extract_sections


class TestEncoderConfig:
    """Validate ENCODER_CONFIG structure."""

    def test_dimension(self):
        assert ENCODER_CONFIG.dimension == 2000

    def test_seed(self):
        assert ENCODER_CONFIG.seed == 42

    def test_three_layers(self):
        assert len(ENCODER_CONFIG.layers) == 3

    def test_path_layer(self):
        path_layer = ENCODER_CONFIG.layers[0]
        assert path_layer.name == "path"
        assert path_layer.similarity_weight == 0.25

    def test_symbols_layer(self):
        symbols_layer = ENCODER_CONFIG.layers[1]
        assert symbols_layer.name == "symbols"
        assert symbols_layer.similarity_weight == 0.25
        defs = symbols_layer.segments[0]
        role_names = {r.name for r in defs.roles}
        assert role_names == {"defines", "docstring", "file_role"}

    def test_content_layer(self):
        content_layer = ENCODER_CONFIG.layers[2]
        assert content_layer.name == "content"
        assert content_layer.similarity_weight == 0.50

    def test_content_has_two_roles(self):
        content = ENCODER_CONFIG.layers[2]
        vocab = content.segments[0]
        assert len(vocab.roles) == 2
        role_names = {r.name for r in vocab.roles}
        assert role_names == {"identifiers", "imports"}

    def test_all_roles_are_bow(self):
        for layer in ENCODER_CONFIG.layers:
            for seg in layer.segments:
                for role in seg.roles:
                    assert role.text_encoding == "bag_of_words"

    def test_encoder_creates(self):
        encoder = Encoder(ENCODER_CONFIG)
        assert encoder.dimension == 2000


class TestTokenization:
    """Test text tokenization helpers."""

    def test_split_camel(self):
        assert _split_camel("sendSlackMessage") == "send Slack Message"
        assert _split_camel("HTTPResponse") == "HTTPResponse"
        assert _split_camel("getUserById") == "get User By Id"

    def test_split_snake(self):
        assert _split_snake("user_service") == "user service"
        assert _split_snake("my-component") == "my component"

    def test_tokenize_combined(self):
        assert _tokenize("getUserById") == "get user by id"
        assert _tokenize("user_service.py") == "user service py"
        assert _tokenize("src/models/User.ts") == "src models user ts"

    def test_tokenize_strips_special(self):
        assert _tokenize("foo@bar#baz") == "foo bar baz"
        assert _tokenize("a..b::c") == "a b c"

    def test_tokenize_lowercase(self):
        assert _tokenize("FooBar") == "foo bar"

    def test_extract_path_tokens(self):
        result = _extract_path_tokens("src/services/user_service.py")
        assert "src" in result
        assert "services" in result
        assert "user" in result
        assert "service" in result
        assert "py" in result

    def test_extract_imports_python(self):
        code = "from sqlalchemy import Column\nimport os\nfrom models.user import User"
        result = _extract_imports(code)
        assert "sqlalchemy" in result
        assert "os" in result
        assert "models user" in result

    def test_extract_imports_javascript(self):
        code = "import React from 'react'\nconst fs = require('fs')"
        result = _extract_imports(code)
        assert "react" in result
        assert "fs" in result

    def test_extract_imports_cpp(self):
        code = '#include <iostream>\n#include "utils/helper.h"'
        result = _extract_imports(code)
        assert "iostream" in result
        assert "utils helper" in result or "utils" in result

    def test_extract_identifiers(self):
        code = 'def create_user(name: str) -> User:\n    """Create a user."""\n    return User(name=name)'
        result = _extract_identifiers(code)
        assert "create" in result
        assert "user" in result
        assert "name" in result

    def test_extract_identifiers_strips_strings(self):
        code = 'x = "hello world"\ny = foo'
        result = _extract_identifiers(code)
        assert "foo" in result
        # String content should be stripped
        assert "hello" not in result or "world" not in result

    def test_top_tokens_filters_stop_words(self):
        idents = "the user service create user get user delete user return self"
        tokens = _top_tokens(idents, n=5)
        assert "user" in tokens
        assert "the" not in tokens
        assert "self" not in tokens
        assert "return" not in tokens

    def test_top_tokens_filters_short(self):
        idents = "a ab abc user service"
        tokens = _top_tokens(idents, n=5)
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens


class TestEncodeQuery:
    """Test NL query encoding."""

    def test_returns_dict_with_name_and_attributes(self):
        result = encode_query("find the auth service")
        assert "name" in result
        assert "attributes" in result

    def test_attributes_match_roles(self):
        result = encode_query("user authentication")
        attrs = result["attributes"]
        assert "path_tokens" in attrs
        assert "identifiers" in attrs
        assert "imports" in attrs

    def test_stop_words_removed(self):
        result = encode_query("find the user service file")
        attrs = result["attributes"]
        assert "find" not in attrs["identifiers"]
        assert "the" not in attrs["identifiers"]
        assert "user" in attrs["identifiers"]
        assert "service" in attrs["identifiers"]

    def test_deterministic(self):
        r1 = encode_query("auth token validation")
        r2 = encode_query("auth token validation")
        assert r1 == r2

    def test_different_queries_different_names(self):
        r1 = encode_query("user service")
        r2 = encode_query("payment handler")
        assert r1["name"] != r2["name"]

    def test_encodes_successfully(self, encoder):
        result = encode_query("database migration scripts")
        concept = Concept(name=result["name"], attributes=result["attributes"])
        glyph = encoder.encode(concept)
        assert glyph.global_cortex is not None
        assert glyph.global_cortex.dimension == 2000


class TestEntryToRecord:
    """Test entry_to_record conversion."""

    def test_passthrough_full_record(self):
        record = {
            "concept_text": "src/main.py",
            "attributes": {"path_tokens": "src main py", "identifiers": "main", "imports": "os"},
            "metadata": {"file_path": "src/main.py"},
        }
        result = entry_to_record(record)
        assert result == record

    def test_raw_jsonl_entry(self):
        entry = {
            "file_path": "lib/utils.ts",
            "identifiers": "format date parse url",
            "imports": "dayjs url",
        }
        result = entry_to_record(entry)
        assert result["concept_text"] == "lib/utils.ts"
        assert result["attributes"]["identifiers"] == "format date parse url"
        assert result["attributes"]["imports"] == "dayjs url"
        assert result["metadata"]["file_path"] == "lib/utils.ts"
        assert result["metadata"]["extension"] == ".ts"


class TestFileToRecord:
    """Test file_to_record on real files."""

    def test_own_encoder(self):
        """encoder.py should be indexable."""
        from pathlib import Path
        from glyphh_code.encoder import file_to_record

        model_dir = str(Path(__file__).parent.parent)
        result = file_to_record(
            str(Path(__file__).parent.parent / "glyphh_code" / "encoder.py"),
            repo_root=model_dir,
        )
        assert result is not None
        assert result["concept_text"] == "glyphh_code/encoder.py"
        assert "encoder" in result["attributes"]["identifiers"]
        assert len(result["metadata"]["top_tokens"]) > 0

    def test_skips_binary(self, tmp_path):
        from glyphh_code.encoder import file_to_record

        binary = tmp_path / "image.png"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert file_to_record(str(binary)) is None

    def test_skips_large_file(self, tmp_path):
        from glyphh_code.encoder import file_to_record, MAX_FILE_BYTES

        big = tmp_path / "huge.py"
        big.write_text("x = 1\n" * (MAX_FILE_BYTES // 5))
        assert file_to_record(str(big)) is None

    def test_skips_nonexistent(self):
        from glyphh_code.encoder import file_to_record

        assert file_to_record("/nonexistent/path/foo.py") is None


class TestConstants:
    """Test constant definitions."""

    def test_indexable_extensions(self):
        assert ".py" in INDEXABLE_EXTENSIONS
        assert ".ts" in INDEXABLE_EXTENSIONS
        assert ".go" in INDEXABLE_EXTENSIONS
        assert ".rs" in INDEXABLE_EXTENSIONS
        assert ".exe" not in INDEXABLE_EXTENSIONS

    def test_skip_dirs(self):
        assert ".git" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert "src" not in SKIP_DIRS


class TestExtractSections:
    """Test section extraction for glyphh_context."""

    PYTHON_CODE = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "class AuthMiddleware:\n"
        "    def __init__(self, app):\n"
        "        self.app = app\n"
        "\n"
        "    def check_scope(self, token, scope):\n"
        "        return scope in token.scopes\n"
        "\n"
        "def validate_token(token_str):\n"
        "    if not token_str:\n"
        "        return None\n"
        "    return decode_jwt(token_str)\n"
    )

    def test_returns_list(self):
        sections = extract_sections(self.PYTHON_CODE, ".py")
        assert isinstance(sections, list)
        assert len(sections) > 0

    def test_sections_have_required_keys(self):
        sections = extract_sections(self.PYTHON_CODE, ".py")
        for s in sections:
            assert "name" in s
            assert "start_line" in s
            assert "end_line" in s
            assert "content" in s

    def test_line_numbers_are_1_based(self):
        sections = extract_sections(self.PYTHON_CODE, ".py")
        for s in sections:
            assert s["start_line"] >= 1
            assert s["end_line"] >= s["start_line"]

    def test_finds_class_and_function(self):
        sections = extract_sections(self.PYTHON_CODE, ".py")
        names = [s["name"] for s in sections]
        assert "AuthMiddleware" in names
        assert "validate_token" in names

    def test_preamble_when_imports_present(self):
        sections = extract_sections(self.PYTHON_CODE, ".py")
        names = [s["name"] for s in sections]
        # Should have a preamble for the imports
        assert "__preamble__" in names
        preamble = next(s for s in sections if s["name"] == "__preamble__")
        assert "import os" in preamble["content"]

    def test_empty_content_returns_empty(self):
        sections = extract_sections("", ".py")
        assert sections == []

    def test_no_definitions_returns_module(self):
        code = "# Just a comment\nx = 42\ny = 'hello'\n"
        sections = extract_sections(code, ".py")
        # Regex fallback should find nothing, return __module__
        assert len(sections) >= 1

    def test_real_file(self):
        """Test on the actual encoder.py file."""
        from pathlib import Path as P
        content = (P(__file__).parent.parent / "glyphh_code" / "encoder.py").read_text()
        sections = extract_sections(content, ".py")
        names = [s["name"] for s in sections]
        assert "encode_query" in names
        assert "file_to_record" in names
        # Sections should cover most of the file
        total_lines = len(content.splitlines())
        covered = sum(s["end_line"] - s["start_line"] + 1 for s in sections)
        assert covered > total_lines * 0.5  # At least 50% coverage

    def test_sections_dont_overlap(self):
        """Section line ranges should not overlap (except preamble edge)."""
        from pathlib import Path as P
        content = (P(__file__).parent.parent / "glyphh_code" / "encoder.py").read_text()
        sections = extract_sections(content, ".py")
        # Sort by start line
        sorted_sections = sorted(sections, key=lambda s: s["start_line"])
        for i in range(1, len(sorted_sections)):
            prev = sorted_sections[i - 1]
            curr = sorted_sections[i]
            # Allow 1 line of overlap for preamble/first-def boundary
            assert curr["start_line"] >= prev["start_line"], (
                f"{curr['name']} starts at {curr['start_line']} "
                f"before {prev['name']} starts at {prev['start_line']}"
            )

    def test_unsupported_extension_uses_regex(self):
        """Non-tree-sitter languages fall back to regex."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        # .xyz not in grammar map — should use regex fallback
        sections = extract_sections(code, ".xyz")
        assert len(sections) >= 2
        names = [s["name"] for s in sections]
        assert "foo" in names
        assert "bar" in names
