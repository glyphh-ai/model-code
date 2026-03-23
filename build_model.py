"""
Build script for the Glyphh Code model.

Unlike static models (toolrouter, pipedream), this model has no fixed exemplar
set. Instead, each user compiles their own codebase via compile.py. This build
script packages the model's encoder config and MCP tool definitions into a
.glyphh artifact for deployment.

Usage:
    python build.py
"""

import json
import sys
from pathlib import Path

from glyphh_code.encoder import ENCODER_CONFIG


def build():
    """Package the code model for deployment."""
    try:
        from glyphh import Encoder
        from glyphh.model.package import GlyphhModel
    except ImportError:
        print("Error: glyphh SDK not installed. Run: pip install glyphh")
        sys.exit(1)

    model_dir = Path(__file__).parent
    encoder = Encoder(ENCODER_CONFIG)

    model = GlyphhModel(
        name="code",
        version="0.1.0",
        encoder_config=ENCODER_CONFIG,
        glyphs=[],
        concepts=[],
        metadata={
            "author": "Glyphh AI",
            "description": "File-level codebase intelligence",
            "category": "search",
        },
    )

    errors = model.validate_completeness()
    if errors:
        print("Validation warnings:")
        for err in errors:
            print(f"  - {err}")

    output_path = model_dir / "code.glyphh"
    model.to_file(str(output_path))
    print(f"Built: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
