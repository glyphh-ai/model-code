#!/usr/bin/env python3
"""Code search benchmark — Glyphh vs bare LLM.

Measures how efficiently Claude finds and acts on files in a codebase
with and without Glyphh. Runs the same queries in two modes:

  Mode A — With Glyphh:
    Claude has glyphh_search as a tool. Calls it, gets ranked files with
    top_tokens and imports. Should find the right file in one tool call.

  Mode B — Without Glyphh (bare LLM):
    Claude has Glob and Read as tools (standard Claude Code). Must scan
    the directory tree and read files to find what it needs.

Metrics:
  - Search accuracy: did the right file appear in results?
  - Tool calls: how many calls to reach the right file?
  - Tokens: input + output tokens consumed
  - Latency: wall-clock time per query
  - Cost: USD per query

Usage:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --model claude-haiku-4-5-20251001
    python benchmark/run_benchmark.py --model claude-sonnet-4-5-20250514
    python benchmark/run_benchmark.py --mode glyphh   # Glyphh only
    python benchmark/run_benchmark.py --mode bare      # bare LLM only
    python benchmark/run_benchmark.py --limit 10       # subset of test cases

Requires:
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# Load .env from bfcl (shared API key)
_BENCHMARK_DIR = Path(__file__).parent
_BFCL_ENV = _BENCHMARK_DIR.parent.parent / "bfcl" / ".env"
if _BFCL_ENV.exists():
    load_dotenv(_BFCL_ENV)
load_dotenv()  # also check local .env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent
TEST_CASES_FILE = BENCHMARK_DIR / "test_cases.json"
RESULTS_DIR = BENCHMARK_DIR / "results"

PRICING = {
    "haiku": {"input": 0.80, "output": 4.00, "label": "Claude Haiku 4.5"},
    "sonnet": {"input": 3.00, "output": 15.00, "label": "Claude Sonnet 4.5"},
    "opus": {"input": 15.00, "output": 75.00, "label": "Claude Opus 4.5"},
}

# MCP endpoint for Glyphh search
MCP_URL_TEMPLATE = "{runtime_url}/{org_id}/{model_id}/mcp"

# Max tool-use loop iterations
MAX_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Glyphh MCP client
# ---------------------------------------------------------------------------

_mcp_session = requests.Session()
_mcp_session.headers.update({
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
})
_jsonrpc_id = 0


def glyphh_search(mcp_url: str, query: str, top_k: int = 5) -> dict:
    """Call glyphh_search via JSON-RPC 2.0."""
    global _jsonrpc_id
    _jsonrpc_id += 1

    resp = _mcp_session.post(
        mcp_url,
        json={
            "jsonrpc": "2.0",
            "id": _jsonrpc_id,
            "method": "tools/call",
            "params": {
                "name": "glyphh_search",
                "arguments": {"query": query, "top_k": top_k},
            },
        },
        timeout=30,
    )
    resp.raise_for_status()

    # Parse SSE or JSON response
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.strip().splitlines():
            if line.startswith("data: "):
                raw = json.loads(line[6:])
                break
        else:
            raw = json.loads(resp.text)
    else:
        raw = resp.json()

    # Unwrap JSON-RPC envelope
    result = raw.get("result", {})
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        return json.loads(content[0]["text"])
    return result


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

GLYPHH_SEARCH_TOOL = {
    "name": "glyphh_search",
    "description": (
        "Search the codebase by natural language query. Returns file paths "
        "with confidence scores, top concepts (top_tokens), and imports."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what you are looking for.",
            },
        },
        "required": ["query"],
    },
}

GLOB_TOOL = {
    "name": "glob",
    "description": "List files matching a glob pattern. Returns file paths.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern like '**/*.py' or 'src/**/*.ts'.",
            },
        },
        "required": ["pattern"],
    },
}

GREP_TOOL = {
    "name": "grep",
    "description": "Search file contents by regex pattern. Returns matching file paths.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for in file contents.",
            },
        },
        "required": ["pattern"],
    },
}

READ_TOOL = {
    "name": "read",
    "description": "Read a file's contents. Returns the file text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["file_path"],
    },
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_GLYPHH = (
    "You are a code assistant with access to a Glyphh codebase index.\n"
    "When asked about code, call glyphh_search to find relevant files.\n"
    "The results include file paths, confidence scores, top_tokens "
    "(key concepts), and imports (dependencies).\n"
    "Use top_tokens and imports to answer without reading the file when possible.\n"
    "Respond with the file path you found and a brief explanation."
)

SYSTEM_BARE = (
    "You are a code assistant working on a Python codebase.\n"
    "You have access to glob (list files), grep (search contents), "
    "and read (read a file).\n"
    "The codebase is rooted at 'src/fastmcp/' with ~760 files.\n"
    "Find the file the user is asking about and respond with "
    "the file path and a brief explanation."
)


# ---------------------------------------------------------------------------
# Simulated bare-LLM tool execution
# ---------------------------------------------------------------------------

# Pre-built file listing for the indexed codebase
_file_tree: list[str] | None = None
_file_contents: dict[str, str] = {}


def _load_file_tree(repo_root: str) -> list[str]:
    """Load the file tree from the repo."""
    global _file_tree
    if _file_tree is not None:
        return _file_tree

    root = Path(repo_root)
    _file_tree = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and not any(
            p in f.parts for p in (".git", "node_modules", "__pycache__", "dist", "build", ".venv")
        ):
            _file_tree.append(str(f.relative_to(root)))
    return _file_tree


def _execute_bare_tool(tool_name: str, tool_input: dict, repo_root: str) -> str:
    """Simulate tool execution for the bare LLM mode."""
    import fnmatch
    import re as _re

    files = _load_file_tree(repo_root)

    if tool_name == "glob":
        pattern = tool_input.get("pattern", "**/*")
        matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
        if len(matches) > 50:
            return json.dumps(matches[:50] + [f"... and {len(matches) - 50} more"])
        return json.dumps(matches)

    elif tool_name == "grep":
        pattern = tool_input.get("pattern", "")
        try:
            regex = _re.compile(pattern, _re.IGNORECASE)
        except _re.error:
            return json.dumps({"error": f"Invalid regex: {pattern}"})

        matches = []
        root = Path(repo_root)
        for f in files:
            fp = root / f
            if fp.suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".yaml", ".json"):
                try:
                    text = _file_contents.get(f)
                    if text is None:
                        text = fp.read_text(errors="ignore")
                        _file_contents[f] = text
                    if regex.search(text):
                        matches.append(f)
                except Exception:
                    pass
            if len(matches) >= 20:
                break
        return json.dumps(matches)

    elif tool_name == "read":
        file_path = tool_input.get("file_path", "")
        root = Path(repo_root)
        fp = root / file_path
        if not fp.exists():
            return json.dumps({"error": f"File not found: {file_path}"})
        try:
            text = fp.read_text(errors="ignore")
            # Truncate to ~4K chars to simulate realistic read
            if len(text) > 4000:
                text = text[:4000] + "\n... (truncated)"
            return text
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# Token / cost counting
# ---------------------------------------------------------------------------

def _model_tier(model: str) -> str:
    if "haiku" in model:
        return "haiku"
    if "sonnet" in model:
        return "sonnet"
    return "opus"


def _compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    prices = PRICING[_model_tier(model)]
    return (input_tokens / 1_000_000) * prices["input"] + \
           (output_tokens / 1_000_000) * prices["output"]


# ---------------------------------------------------------------------------
# Run a single test case
# ---------------------------------------------------------------------------

def run_glyphh_test(
    client: anthropic.Anthropic,
    model: str,
    mcp_url: str,
    test_case: dict,
) -> dict:
    """Run a test case with Glyphh search."""
    query = test_case["query"]
    expected = test_case.get("expected_file") or test_case.get("target_file", "")
    top_k_required = test_case.get("expected_in_top_k", 5)

    t0 = time.perf_counter()
    total_input = 0
    total_output = 0
    tool_calls = 0
    found_file = None
    glyphh_results = []

    messages = [{"role": "user", "content": query}]

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_GLYPHH,
            messages=messages,
            tools=[GLYPHH_SEARCH_TOOL],
        )

        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        tool_results = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls += 1
                search_query = block.input.get("query", query)
                result = glyphh_search(mcp_url, search_query, top_k=top_k_required)
                glyphh_results.append(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if not tool_results:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Check if expected file was found in any Glyphh result
    all_files = []
    for r in glyphh_results:
        ft = r.get("fact_tree", {})
        for child in ft.get("children", []):
            sample = child.get("data_sample", {})
            f = sample.get("file") or child.get("description", "")
            all_files.append(f)

    found = expected in all_files
    rank = all_files.index(expected) + 1 if found else -1

    return {
        "id": test_case["id"],
        "type": test_case["type"],
        "query": query,
        "expected_file": expected,
        "found": found,
        "rank": rank,
        "tool_calls": tool_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cost_usd": round(_compute_cost(total_input, total_output, model), 6),
        "latency_ms": round(elapsed_ms, 1),
        "files_returned": all_files[:10],
        "response": " ".join(text_parts)[:200] if text_parts else "",
    }


def run_bare_test(
    client: anthropic.Anthropic,
    model: str,
    repo_root: str,
    test_case: dict,
) -> dict:
    """Run a test case without Glyphh (bare LLM with glob/grep/read)."""
    query = test_case["query"]
    expected = test_case.get("expected_file") or test_case.get("target_file", "")

    t0 = time.perf_counter()
    total_input = 0
    total_output = 0
    tool_calls = 0
    files_read = []
    text_parts = []

    messages = [{"role": "user", "content": query}]
    tools = [GLOB_TOOL, GREP_TOOL, READ_TOOL]

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_BARE,
            messages=messages,
            tools=tools,
        )

        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        tool_results = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls += 1
                result_text = _execute_bare_tool(
                    block.name, block.input, repo_root,
                )
                if block.name == "read":
                    files_read.append(block.input.get("file_path", ""))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        if not tool_results:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Check if Claude mentioned or read the expected file
    all_text = " ".join(text_parts)
    found = expected in all_text or expected in files_read

    return {
        "id": test_case["id"],
        "type": test_case["type"],
        "query": query,
        "expected_file": expected,
        "found": found,
        "rank": -1,  # Not applicable for bare mode
        "tool_calls": tool_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cost_usd": round(_compute_cost(total_input, total_output, model), 6),
        "latency_ms": round(elapsed_ms, 1),
        "files_read": files_read,
        "response": all_text[:200],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(label: str, results: list[dict], model: str):
    """Print a summary table for a set of results."""
    total = len(results)
    found = sum(1 for r in results if r["found"])
    accuracy = found / total * 100 if total else 0

    total_tokens = sum(r["total_tokens"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_tool_calls = sum(r["tool_calls"] for r in results) / total if total else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0
    avg_tokens = total_tokens / total if total else 0

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Model: {PRICING[_model_tier(model)]['label']} ({model})")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {found}/{total} ({accuracy:.1f}%)")
    print(f"  Avg tool calls: {avg_tool_calls:.1f}")
    print(f"  Avg tokens:     {avg_tokens:.0f}")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"  Total cost:     ${total_cost:.4f}")
    print()

    for test_type, type_results in sorted(by_type.items()):
        n = len(type_results)
        f = sum(1 for r in type_results if r["found"])
        tc = sum(r["tool_calls"] for r in type_results) / n
        tk = sum(r["total_tokens"] for r in type_results) / n
        print(f"  {test_type:12s}  {f}/{n} correct  avg {tc:.1f} calls  avg {tk:.0f} tokens")

    print()

    # Show failures
    failures = [r for r in results if not r["found"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            print(f"    {r['id']}: expected {r['expected_file']}")
            if r.get("files_returned"):
                print(f"      got: {r['files_returned'][:3]}")
            elif r.get("files_read"):
                print(f"      read: {r['files_read'][:3]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Code search benchmark")
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Anthropic model ID (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--mode", choices=["both", "glyphh", "bare"], default="both",
        help="Which mode to run (default: both)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of test cases (0 = all)",
    )
    parser.add_argument(
        "--repo-root", default="/Users/timmetim/development/a-test/fastmcp",
        help="Path to the indexed repo (for bare mode)",
    )
    parser.add_argument(
        "--types", nargs="+", default=None,
        help="Only run specific test types (search, edit, debug, understand)",
    )
    args = parser.parse_args()

    # Load test cases
    with open(TEST_CASES_FILE) as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    runtime_url = data["runtime_url"]
    org_id = data["org_id"]
    model_id = data["model_id"]
    mcp_url = MCP_URL_TEMPLATE.format(
        runtime_url=runtime_url, org_id=org_id, model_id=model_id,
    )

    if args.types:
        test_cases = [t for t in test_cases if t["type"] in args.types]
    if args.limit > 0:
        test_cases = test_cases[:args.limit]

    print(f"Code Search Benchmark")
    print(f"Model:      {args.model}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Mode:       {args.mode}")
    print(f"MCP URL:    {mcp_url}")
    print()

    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    tier = _model_tier(args.model)

    # --- Glyphh mode ---
    if args.mode in ("both", "glyphh"):
        print("Running Glyphh mode...")
        glyphh_results = []
        for i, tc in enumerate(test_cases):
            try:
                r = run_glyphh_test(client, args.model, mcp_url, tc)
                glyphh_results.append(r)
                status = "✓" if r["found"] else "✗"
                print(f"  [{i+1}/{len(test_cases)}] {status} {tc['id']} "
                      f"({r['tool_calls']} calls, {r['total_tokens']} tokens, "
                      f"{r['latency_ms']:.0f}ms)")
            except Exception as e:
                print(f"  [{i+1}/{len(test_cases)}] ERROR {tc['id']}: {e}")
                glyphh_results.append({
                    "id": tc["id"], "type": tc["type"], "query": tc["query"],
                    "expected_file": tc.get("expected_file", tc.get("target_file", "")),
                    "found": False, "rank": -1, "tool_calls": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cost_usd": 0, "latency_ms": 0, "error": str(e),
                })

        out_path = RESULTS_DIR / f"glyphh_{tier}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "glyphh", "results": glyphh_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("WITH GLYPHH", glyphh_results, args.model)

    # --- Bare LLM mode ---
    if args.mode in ("both", "bare"):
        print("Running bare LLM mode...")
        bare_results = []
        for i, tc in enumerate(test_cases):
            try:
                r = run_bare_test(client, args.model, args.repo_root, tc)
                bare_results.append(r)
                status = "✓" if r["found"] else "✗"
                print(f"  [{i+1}/{len(test_cases)}] {status} {tc['id']} "
                      f"({r['tool_calls']} calls, {r['total_tokens']} tokens, "
                      f"{r['latency_ms']:.0f}ms)")
            except Exception as e:
                print(f"  [{i+1}/{len(test_cases)}] ERROR {tc['id']}: {e}")
                bare_results.append({
                    "id": tc["id"], "type": tc["type"], "query": tc["query"],
                    "expected_file": tc.get("expected_file", tc.get("target_file", "")),
                    "found": False, "rank": -1, "tool_calls": 0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cost_usd": 0, "latency_ms": 0, "error": str(e),
                })

        out_path = RESULTS_DIR / f"bare_{tier}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "bare", "results": bare_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("WITHOUT GLYPHH (bare LLM)", bare_results, args.model)

    # --- Comparison ---
    if args.mode == "both":
        print("=" * 60)
        print("  HEAD-TO-HEAD COMPARISON")
        print("=" * 60)

        g = glyphh_results
        b = bare_results
        g_found = sum(1 for r in g if r["found"])
        b_found = sum(1 for r in b if r["found"])
        g_tokens = sum(r["total_tokens"] for r in g)
        b_tokens = sum(r["total_tokens"] for r in b)
        g_cost = sum(r["cost_usd"] for r in g)
        b_cost = sum(r["cost_usd"] for r in b)
        g_calls = sum(r["tool_calls"] for r in g)
        b_calls = sum(r["tool_calls"] for r in b)

        n = len(test_cases)
        print(f"  {'':20s} {'Glyphh':>10s} {'Bare LLM':>10s} {'Ratio':>10s}")
        print(f"  {'Accuracy':20s} {g_found:>5d}/{n:<4d} {b_found:>5d}/{n:<4d}")
        print(f"  {'Avg tokens':20s} {g_tokens/n:>10.0f} {b_tokens/n:>10.0f} {b_tokens/max(g_tokens,1):>9.1f}x")
        print(f"  {'Avg tool calls':20s} {g_calls/n:>10.1f} {b_calls/n:>10.1f} {b_calls/max(g_calls,1):>9.1f}x")
        print(f"  {'Total cost':20s} ${g_cost:>9.4f} ${b_cost:>9.4f} {b_cost/max(g_cost,0.0001):>9.1f}x")
        print()


if __name__ == "__main__":
    main()
