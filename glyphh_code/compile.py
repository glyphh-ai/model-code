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

from glyphh_code.encoder import SKIP_DIRS, INDEXABLE_EXTENSIONS, MAX_FILE_BYTES, file_to_record


DEFAULT_RUNTIME_URL = "http://localhost:8002"
DEFAULT_ORG_ID = "local-dev-org"
DEFAULT_MODEL_ID = "code"
BATCH_SIZE = 50

GLYPHH_CONFIG_FILE = Path.home() / ".glyphh" / "config.json"


def _load_cli_config() -> dict:
    """Load the Glyphh CLI config (~/.glyphh/config.json)."""
    if GLYPHH_CONFIG_FILE.exists():
        try:
            return json.loads(GLYPHH_CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _resolve_token(explicit: str | None, runtime_url: str) -> str | None:
    """Return the auth token to use.

    Priority: explicit --token / GLYPHH_TOKEN →
              runtime_tokens[endpoint] → runtime_token → access_token.
    Only reads the CLI session for remote (non-localhost) URLs.
    """
    if explicit:
        return explicit
    from urllib.parse import urlparse
    host = urlparse(runtime_url).hostname
    if host in ("localhost", "127.0.0.1", "::1"):
        return None
    config = _load_cli_config()

    # Per-endpoint token (matches CLI's resolve_runtime_token priority)
    runtime_tokens = config.get("runtime_tokens", {})
    if isinstance(runtime_tokens, dict):
        endpoint_token = runtime_tokens.get(runtime_url, "").strip()
        if endpoint_token:
            print(f"Using token from CLI session (runtime_tokens[{runtime_url}])")
            return endpoint_token

    # Legacy singular runtime_token
    runtime_token = config.get("runtime_token", "").strip()
    if runtime_token:
        print("Using token from CLI session (runtime_token)")
        return runtime_token

    # Platform JWT (access_token) — runtime accepts both
    access_token = config.get("access_token", "").strip()
    if access_token:
        print("Using token from CLI session (access_token)")
        return access_token

    print("Warning: no --token provided and no CLI session found. Run: glyphh auth login")
    return None


def _resolve_org_id(explicit: str, runtime_url: str) -> str:
    """Return the org ID to use.

    Priority: explicit --org-id / GLYPHH_ORG_ID → CLI session org_id.
    Falls back to DEFAULT_ORG_ID for localhost.
    """
    if explicit != DEFAULT_ORG_ID:
        return explicit
    from urllib.parse import urlparse
    host = urlparse(runtime_url).hostname
    if host in ("localhost", "127.0.0.1", "::1"):
        return DEFAULT_ORG_ID
    config = _load_cli_config()
    org_id = config.get("user", {}).get("org_id")
    if org_id:
        print(f"Using org_id from CLI session: {org_id}")
        return org_id
    return DEFAULT_ORG_ID


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


def get_changed_files_range(repo_root: str, diff_range: str) -> list[str]:
    """Get files changed in a commit range (e.g. 'abc123..HEAD').

    Used after git pull/merge/rebase to find all files that changed
    between the pre-operation HEAD and the current HEAD.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", diff_range],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        print(f"Warning: git diff failed for range {diff_range}: {result.stderr.strip()}")
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
    diff_range: str | None = None,
    diff_repo: str | None = None,
    dry_run: bool = False,
) -> int:
    """Compile a repository into the Glyphh code index.

    Args:
        diff_repo: When set, run git diff in this repo instead of repo_root.
                   Changed file paths are remapped to repo_root-relative paths.
                   Used when a commit lands in a child repo / submodule.

    Uses tree-sitter AST extraction (with regex fallback) for the symbols
    layer — no LLM required. Fully deterministic.
    """
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        print(f"Error: {repo_root} is not a directory")
        sys.exit(1)

    print(f"Compiling: {repo_root}")

    # Determine file list
    if diff:
        files = get_changed_files(repo_root, diff)
        print(f"Mode: diff {diff} ({len(files)} changed files)")
    elif diff_range:
        diff_dir = os.path.abspath(diff_repo) if diff_repo else repo_root
        files = get_changed_files_range(diff_dir, diff_range)
        if diff_repo and diff_dir != repo_root:
            remapped = []
            for f in files:
                if os.path.exists(f):
                    remapped.append(f)
                else:
                    rel = os.path.relpath(f, diff_dir)
                    full = os.path.join(diff_dir, rel)
                    if os.path.exists(full):
                        remapped.append(full)
            files = remapped
        print(f"Mode: diff-range {diff_range} ({len(files)} changed files)")
    elif incremental:
        # If diff_repo is set, diff there and remap paths to repo_root
        diff_dir = os.path.abspath(diff_repo) if diff_repo else repo_root
        files = get_changed_files(diff_dir, "HEAD")
        if diff_repo and diff_dir != repo_root:
            # Remap: files are absolute under diff_dir, which is a subdir of repo_root
            remapped = []
            for f in files:
                if os.path.exists(f):
                    remapped.append(f)
                else:
                    # Try relative to diff_dir, then map into repo_root
                    rel = os.path.relpath(f, diff_dir)
                    full = os.path.join(diff_dir, rel)
                    if os.path.exists(full):
                        remapped.append(full)
            files = remapped
            print(f"Mode: incremental via {diff_dir} ({len(files)} changed files)")
        else:
            print(f"Mode: incremental ({len(files)} changed files)")
    else:
        files = walk_repo(repo_root)
        print(f"Mode: full ({len(files)} candidate files)")

    # Generate records (AST extraction happens inside file_to_record)
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
            defines_preview = r.get("attributes", {}).get("defines", "")[:60]
            print(f"  {r['concept_text']:50s}  {defines_preview or ' '.join(tokens)}")
        if len(records) > 20:
            print(f"  ... and {len(records) - 20} more")
        return len(records), []

    if not records:
        print("Nothing to index.")
        return 0, []

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
            "symbols": {
                "definitions": {
                    "defines": attrs.get("defines", ""),
                    "docstring": attrs.get("docstring", ""),
                    "file_role": attrs.get("file_role", "source"),
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

    # Post all records in a single request — the listener processes internal
    # batches sequentially within one job, avoiding SQLite write contention.
    start = time.time()
    result = post_to_listener(hierarchical_records, runtime_url, org_id, model_id, token)
    job_id = result.get("job_id", "?")
    print(f"  {len(hierarchical_records)} records → job {job_id}")

    elapsed = time.time() - start
    print(f"Done: {len(records)} files indexed in {elapsed:.1f}s")
    return len(records), [job_id] if job_id != "?" else []


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
    parser.add_argument("--diff-range", type=str, help="Compile changes in a commit range (e.g. ORIG_HEAD..HEAD)")
    parser.add_argument("--diff-repo", type=str, help="Git repo to diff (when commit is in a child repo/submodule)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = _resolve_token(args.token, args.runtime_url)
    org_id = _resolve_org_id(args.org_id, args.runtime_url)

    compile_repo(
        repo_root=args.repo,
        runtime_url=args.runtime_url,
        org_id=org_id,
        model_id=args.model_id,
        token=token,
        incremental=args.incremental,
        diff=args.diff,
        diff_range=args.diff_range,
        diff_repo=args.diff_repo,
        dry_run=args.dry_run,
    )
