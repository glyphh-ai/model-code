"""
Build pipeline for glyphh-code.

Walks a repository, encodes each source file into an HDC vector,
and posts records to the Glyphh runtime listener for storage in pgvector.

Usage:
    # Full compile (all indexable files)
    python compile.py /path/to/repo --runtime-url http://localhost:8002

    # Incremental (changed files since last commit)
    python compile.py /path/to/repo --incremental

    # Specific commit diff
    python compile.py /path/to/repo --diff abc123

    # Dry run (show what would be indexed)
    python compile.py /path/to/repo --dry-run

Requires: requests (pip install requests)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Allow importing from this directory
sys.path.insert(0, str(Path(__file__).parent))

from encoder import SKIP_DIRS, INDEXABLE_EXTENSIONS, file_to_record


DEFAULT_RUNTIME_URL = "http://localhost:8002"
DEFAULT_ORG_ID = "local-dev-org"
DEFAULT_MODEL_ID = "code"
BATCH_SIZE = 50


def walk_repo(root: str) -> list[str]:
    """Walk a repo directory and return all indexable file paths."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if Path(filename).suffix in INDEXABLE_EXTENSIONS:
                files.append(full_path)
    return files


def get_changed_files(repo_root: str, commit_hash: str = "HEAD") -> list[str]:
    """Get files changed in a specific commit."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{commit_hash}^", commit_hash],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        print(f"Warning: git diff failed: {result.stderr.strip()}")
        return []
    return [
        os.path.join(repo_root, f)
        for f in result.stdout.strip().split("\n")
        if f
    ]


def post_to_listener(
    records: list[dict],
    runtime_url: str,
    org_id: str,
    model_id: str,
    token: str | None = None,
) -> dict:
    """POST a batch of records to the runtime listener endpoint."""
    import requests

    url = f"{runtime_url}/{org_id}/{model_id}/listener"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(
        url,
        json={"records": records, "batch_size": BATCH_SIZE},
        headers=headers,
        timeout=120,
    )

    if resp.status_code not in (200, 201, 202):
        print(f"Error: listener returned {resp.status_code}: {resp.text}")
        sys.exit(1)

    return resp.json()


def compile_repo(
    repo_root: str,
    runtime_url: str = DEFAULT_RUNTIME_URL,
    org_id: str = DEFAULT_ORG_ID,
    model_id: str = DEFAULT_MODEL_ID,
    token: str | None = None,
    incremental: bool = False,
    diff: str | None = None,
    dry_run: bool = False,
) -> int:
    """Compile a repository into the Glyphh code index."""
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        print(f"Error: {repo_root} is not a directory")
        sys.exit(1)

    print(f"Compiling: {repo_root}")

    # Determine file list
    if diff:
        files = get_changed_files(repo_root, diff)
        print(f"Mode: diff {diff} ({len(files)} changed files)")
    elif incremental:
        files = get_changed_files(repo_root, "HEAD")
        print(f"Mode: incremental ({len(files)} changed files)")
    else:
        files = walk_repo(repo_root)
        print(f"Mode: full ({len(files)} candidate files)")

    # Generate records
    records = []
    skipped = 0
    for file_path in files:
        record = file_to_record(file_path, repo_root)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"Encoded: {len(records)} files ({skipped} skipped)")

    if dry_run:
        for r in records[:20]:
            meta = r.get("metadata", {})
            tokens = meta.get("top_tokens", [])[:5]
            print(f"  {r['concept_text']:50s}  {' '.join(tokens)}")
        if len(records) > 20:
            print(f"  ... and {len(records) - 20} more")
        return len(records)

    if not records:
        print("Nothing to index.")
        return 0

    # Send records in hierarchical format matching the encoder config.
    # The listener auto-maps layer/segment/role structure for encoding
    # and preserves concept_text and metadata on the stored glyph.
    hierarchical_records = []
    for r in records:
        attrs = r.get("attributes", {})
        hier = {
            "path": {
                "location": {
                    "path_tokens": attrs.get("path_tokens", ""),
                },
            },
            "content": {
                "vocabulary": {
                    "identifiers": attrs.get("identifiers", ""),
                    "imports": attrs.get("imports", ""),
                },
            },
            "concept_text": r.get("concept_text", ""),
            "metadata": r.get("metadata", {}),
        }
        hierarchical_records.append(hier)

    # Post to runtime in batches
    start = time.time()
    for i in range(0, len(hierarchical_records), BATCH_SIZE):
        batch = hierarchical_records[i : i + BATCH_SIZE]
        result = post_to_listener(batch, runtime_url, org_id, model_id, token)
        job_id = result.get("job_id", "?")
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} records → job {job_id}")

    elapsed = time.time() - start
    print(f"Done: {len(records)} files indexed in {elapsed:.1f}s")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile a repository into the Glyphh code index"
    )
    parser.add_argument("repo", help="Path to the repository root")
    parser.add_argument(
        "--runtime-url",
        default=os.environ.get("GLYPHH_RUNTIME_URL", DEFAULT_RUNTIME_URL),
        help=f"Runtime URL (default: {DEFAULT_RUNTIME_URL})",
    )
    parser.add_argument(
        "--org-id",
        default=os.environ.get("GLYPHH_ORG_ID", DEFAULT_ORG_ID),
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GLYPHH_MODEL_ID", DEFAULT_MODEL_ID),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GLYPHH_TOKEN"),
        help="Auth token for the runtime",
    )
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--diff", type=str, help="Compile changes from a specific commit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compile_repo(
        repo_root=args.repo,
        runtime_url=args.runtime_url,
        org_id=args.org_id,
        model_id=args.model_id,
        token=args.token,
        incremental=args.incremental,
        diff=args.diff,
        dry_run=args.dry_run,
    )
