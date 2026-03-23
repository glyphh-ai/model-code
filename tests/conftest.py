"""Shared test fixtures for the Glyphh Code model."""

import sys
from pathlib import Path

import pytest

# Ensure glyphh SDK is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "glyphh-runtime"))

from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh_code.encoder import (
    ENCODER_CONFIG,
    encode_query,
    entry_to_record,
    file_to_record,
    _tokenize,
    _extract_path_tokens,
    _extract_imports,
    _extract_identifiers,
    _top_tokens,
)


@pytest.fixture
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture
def sample_file_record():
    """A record representing a typical Python service file."""
    return {
        "concept_text": "src/services/user_service.py",
        "attributes": {
            "path_tokens": "src services user service py",
            "identifiers": (
                "user service create user get user by id "
                "update user delete user hash password "
                "validate email send verification"
            ),
            "imports": (
                "sqlalchemy fastapi pydantic bcrypt "
                "models user schemas user service"
            ),
        },
        "metadata": {
            "file_path": "src/services/user_service.py",
            "extension": ".py",
            "file_size": 3200,
            "top_tokens": [
                "user", "service", "create", "password",
                "email", "validate", "hash", "delete",
            ],
            "imports": ["sqlalchemy", "fastapi", "pydantic", "bcrypt"],
        },
    }


@pytest.fixture
def sample_test_record():
    """A record representing a test file."""
    return {
        "concept_text": "tests/test_user_service.py",
        "attributes": {
            "path_tokens": "tests test user service py",
            "identifiers": (
                "test create user test get user test delete user "
                "test validate email mock database fixture user"
            ),
            "imports": "pytest mock services user service models user",
        },
        "metadata": {
            "file_path": "tests/test_user_service.py",
            "extension": ".py",
            "file_size": 2800,
            "top_tokens": [
                "test", "user", "service", "create",
                "delete", "validate", "mock", "fixture",
            ],
            "imports": ["pytest", "mock", "services"],
        },
    }


@pytest.fixture
def sample_config_record():
    """A record representing a config/schema file."""
    return {
        "concept_text": "src/models/user.py",
        "attributes": {
            "path_tokens": "src models user py",
            "identifiers": (
                "user model base email first name last name "
                "password hash created updated column string "
                "datetime boolean relationship"
            ),
            "imports": "sqlalchemy column string datetime boolean relationship",
        },
        "metadata": {
            "file_path": "src/models/user.py",
            "extension": ".py",
            "file_size": 1200,
            "top_tokens": [
                "user", "model", "email", "name",
                "password", "column", "string", "datetime",
            ],
            "imports": ["sqlalchemy"],
        },
    }
