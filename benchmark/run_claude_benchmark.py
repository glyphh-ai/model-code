#!/usr/bin/env python3
"""Claude Code benchmark — Glyphh + LLM vs bare LLM.

Runs real Claude Code sessions against a target repo. Measures the actual
tool calls, tokens, and cost Claude Code uses to complete each task.

Two test types:
  blast_radius  — "what breaks if I edit X?" (Glyphh's strength)
  semantic      — conceptual queries with no exact string match

Two modes:
  combined — Claude Code with Glyphh MCP server + grep/glob/read
  bare     — Claude Code without Glyphh (grep/glob/read only)

Usage:
    python benchmark/run_claude_benchmark.py                  # both modes
    python benchmark/run_claude_benchmark.py --mode combined  # with Glyphh
    python benchmark/run_claude_benchmark.py --mode bare      # without Glyphh
    python benchmark/run_claude_benchmark.py --limit 5        # subset
    python benchmark/run_claude_benchmark.py --model sonnet   # use Sonnet
    python benchmark/run_claude_benchmark.py --types blast_radius semantic
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from functools import partial
from dotenv import load_dotenv

# Force unbuffered output
print = partial(print, flush=True)

_BENCHMARK_DIR = Path(__file__).parent
_BFCL_ENV = _BENCHMARK_DIR.parent.parent / "bfcl" / ".env"
if _BFCL_ENV.exists():
    load_dotenv(_BFCL_ENV)
load_dotenv()

TEST_CASES_FILE = _BENCHMARK_DIR / "test_cases.json"
RESULTS_DIR = _BENCHMARK_DIR / "results"
REPO_ROOT = "/Users/timmetim/development/a-test/fastmcp"

# MCP config for Glyphh — passed via --mcp-config
MCP_CONFIG = {
    "mcpServers": {
        "glyphh": {
            "type": "http",
            "url": "http://localhost:8002/local-dev-org/code/mcp",
        }
    }
}
MCP_CONFIG_FILE = _BENCHMARK_DIR / ".mcp-benchmark.json"

STATUS_FILE = RESULTS_DIR / "status.json"

# --- Prompts ---
# Blast radius: identify what breaks if a file is edited.
PROMPT_BLAST = (
    "TASK: The user is about to edit a file. Identify ALL other files that "
    "could break or need coordinated changes. Do NOT edit anything.\n"
    "Respond with a list of affected file paths (one per line), then brief "
    "reasoning for each. Focus on direct dependents, not transitive.\n"
    "Do NOT ask clarifying questions — just analyze and respond."
)

# Semantic: find files related to a concept.
PROMPT_SEMANTIC = (
    "TASK: Find the source files most relevant to the user's question. "
    "List the top file paths (one per line) then a brief explanation of each.\n"
    "Do NOT ask clarifying questions — just find and list the files."
)

# Glyphh-specific guidance — tells Claude when to use each tool.
GLYPHH_GUIDANCE = (
    "You have access to Glyphh MCP tools in addition to grep/glob/read.\n"
    "Use Grep/Glob for navigation — finding where something is defined, "
    "exact string matches, symbol lookups.\n"
    "Use glyphh_search for semantic queries that Grep cannot answer.\n"
    "Use glyphh_related before editing any file to understand blast radius — "
    "it returns semantically similar files that may need coordinated changes. "
    "There is no Grep equivalent for this.\n"
    "IMPORTANT: Always pass detail='minimal' to glyphh_search and glyphh_related "
    "to keep responses lightweight (file path + confidence only).\n"
)


def _get_prompt(test_type: str, with_glyphh: bool) -> str:
    """Return the appropriate system prompt for the test type and mode."""
    base = {
        "blast_radius": PROMPT_BLAST,
        "semantic": PROMPT_SEMANTIC,
    }.get(test_type, PROMPT_SEMANTIC)

    if with_glyphh:
        return GLYPHH_GUIDANCE + "\n" + base
    return base


def _get_budget(test_type: str) -> float:
    """Return max budget per test type."""
    return {
        "blast_radius": 0.30,
        "semantic": 0.25,
    }.get(test_type, 0.25)


def _compute_stats(results: list[dict]) -> dict:
    """Compute running stats for a list of results."""
    n = len(results)
    if n == 0:
        return {}
    found = sum(1 for r in results if r["found"])
    return {
        "completed": n,
        "accuracy": round(found / n * 100, 1),
        "found": found,
        "avg_tokens": round(sum(r["total_tokens"] for r in results) / n),
        "avg_turns": round(sum(r["num_turns"] for r in results) / n, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
    }


def write_status(
    model: str,
    total_cases: int,
    bare_results: list[dict] | None = None,
    combined_results: list[dict] | None = None,
    phase: str = "running",
):
    """Write live status JSON after each test case."""
    status: dict = {
        "model": model,
        "total_cases": total_cases,
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if bare_results is not None:
        status["bare"] = {
            "stats": _compute_stats(bare_results),
            "results": bare_results,
        }

    if combined_results is not None:
        status["combined"] = {
            "stats": _compute_stats(combined_results),
            "results": combined_results,
        }

    # Comparison if both exist
    if bare_results and combined_results:
        bs = _compute_stats(bare_results)
        cs = _compute_stats(combined_results)
        b_tok = max(bs.get("avg_tokens", 1), 1)
        b_turns = max(bs.get("avg_turns", 1), 0.1)
        b_cost = max(bs.get("total_cost_usd", 0.0001), 0.0001)
        status["comparison"] = {
            "token_savings_pct": round((1 - cs.get("avg_tokens", 0) / b_tok) * 100),
            "turn_savings_pct": round((1 - cs.get("avg_turns", 0) / b_turns) * 100),
            "cost_savings_pct": round((1 - cs.get("total_cost_usd", 0) / b_cost) * 100),
            "bare_accuracy": bs.get("accuracy", 0),
            "combined_accuracy": cs.get("accuracy", 0),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def run_claude(
    query: str,
    model: str,
    with_glyphh: bool,
    test_type: str = "navigation",
) -> dict:
    """Run a single Claude Code session and return structured results."""
    prompt = _get_prompt(test_type, with_glyphh)
    budget = _get_budget(test_type)

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--model", model,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--max-budget-usd", str(budget),
    ]

    if with_glyphh:
        MCP_CONFIG_FILE.write_text(json.dumps(MCP_CONFIG))
        cmd.extend(["--mcp-config", str(MCP_CONFIG_FILE)])

    cmd.extend(["--append-system-prompt", prompt])

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            cmd,
            input=query,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=180,
        )

        if result.returncode != 0:
            return {"error": result.stderr.strip()[:200]}

        data = json.loads(result.stdout.strip())
        return data

    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except json.JSONDecodeError:
        return {"error": f"bad json: {result.stdout[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def extract_files_from_result(result_text: str) -> list[str]:
    """Extract file paths from Claude's response."""
    if not result_text:
        return []
    files = []
    for line in result_text.strip().split("\n"):
        line = line.strip().strip("`").strip("*").strip("- ").strip()
        # Remove leading prefixes
        for prefix in ("File:", "Path:", "file:", "path:"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        # Check if line looks like a file path
        if "/" in line and not line.startswith("(") and len(line) < 200:
            # Extract just the path part (stop at whitespace or common delimiters)
            path = line.split(" ")[0].split("\t")[0].strip("`").strip("*")
            # Remove markdown link syntax [text](path)
            if "](" in path:
                path = path.split("](")[-1].rstrip(")")
            if path.count("/") >= 1 and not path.startswith("http"):
                files.append(path)
    return files


def evaluate_result(test_case: dict, result_text: str) -> bool:
    """Evaluate whether the result is correct based on test type."""
    test_type = test_case["type"]

    if test_type in ("blast_radius", "semantic"):
        expected_files = test_case.get("expected_files", [])
        min_expected = test_case.get("min_expected", 1)
        found_count = 0
        for ef in expected_files:
            if ef in result_text:
                found_count += 1
        return found_count >= min_expected

    return False


def run_test(
    test_case: dict,
    model: str,
    with_glyphh: bool,
) -> dict:
    """Run a single test case and return metrics."""
    query = test_case["query"]
    test_type = test_case["type"]

    expected_display = f"{test_case.get('min_expected', 1)}+ of {len(test_case.get('expected_files', []))} files"

    t0 = time.perf_counter()
    raw = run_claude(query, model, with_glyphh, test_type)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if "error" in raw:
        return {
            "id": test_case["id"],
            "type": test_type,
            "query": query,
            "expected": expected_display,
            "found": False,
            "result_files": [],
            "num_turns": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "latency_ms": round(elapsed_ms, 1),
            "error": raw["error"],
        }

    result_text = raw.get("result", "")
    result_files = extract_files_from_result(result_text)
    found = evaluate_result(test_case, result_text)
    cost = raw.get("total_cost_usd", 0)
    num_turns = raw.get("num_turns", 0)
    duration = raw.get("duration_ms", elapsed_ms)

    usage = raw.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    total_tokens = input_tokens + output_tokens + cache_read + cache_create

    return {
        "id": test_case["id"],
        "type": test_type,
        "query": query,
        "expected": expected_display,
        "found": found,
        "result_files": result_files[:10],
        "num_turns": num_turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_create_tokens": cache_create,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(duration, 1),
        "response_preview": result_text[:300] if result_text else "",
    }


def print_summary(label: str, results: list[dict], model: str):
    """Print summary table."""
    total = len(results)
    found = sum(1 for r in results if r["found"])
    accuracy = found / total * 100 if total else 0

    total_tokens = sum(r["total_tokens"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_turns = sum(r["num_turns"] for r in results) / total if total else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0
    avg_tokens = total_tokens / total if total else 0

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Model: {model}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:       {found}/{total} ({accuracy:.1f}%)")
    print(f"  Avg turns:      {avg_turns:.1f}")
    print(f"  Avg tokens:     {avg_tokens:.0f}")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"  Total cost:     ${total_cost:.4f}")
    print()

    for test_type, type_results in sorted(by_type.items()):
        n = len(type_results)
        f = sum(1 for r in type_results if r["found"])
        tc = sum(r["num_turns"] for r in type_results) / n
        tk = sum(r["total_tokens"] for r in type_results) / n
        cost = sum(r["cost_usd"] for r in type_results)
        print(f"  {test_type:15s}  {f}/{n} correct  avg {tc:.1f} turns  "
              f"avg {tk:.0f} tok  ${cost:.4f}")

    print()

    failures = [r for r in results if not r["found"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            err = f"  error: {r['error']}" if r.get("error") else ""
            print(f"    {r['id']}: expected {r['expected']}")
            got = r['result_files'][:3] if r.get('result_files') else ['(none)']
            print(f"      got: {', '.join(got)}{err}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Claude Code benchmark")
    parser.add_argument(
        "--model", default="haiku",
        help="Model alias: haiku, sonnet, opus (default: haiku)",
    )
    parser.add_argument(
        "--mode", choices=["both", "combined", "bare"], default="both",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--types", nargs="+", default=None,
        help="Only run specific test types: blast_radius, semantic",
    )
    args = parser.parse_args()

    with open(TEST_CASES_FILE) as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    if args.types:
        test_cases = [t for t in test_cases if t["type"] in args.types]
    if args.limit > 0:
        test_cases = test_cases[:args.limit]

    # Count by type
    type_counts = {}
    for tc in test_cases:
        type_counts[tc["type"]] = type_counts.get(tc["type"], 0) + 1

    print(f"Claude Code Benchmark")
    print(f"Model:      {args.model}")
    print(f"Test cases: {len(test_cases)} ({', '.join(f'{v} {k}' for k, v in type_counts.items())})")
    print(f"Mode:       {args.mode}")
    print(f"Repo:       {REPO_ROOT}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    bare_results = None
    combined_results = None

    # --- Bare LLM ---
    if args.mode in ("both", "bare"):
        print("Running bare LLM mode (no Glyphh)...")
        bare_results = []
        for i, tc in enumerate(test_cases):
            r = run_test(tc, args.model, with_glyphh=False)
            bare_results.append(r)
            mark = "✓" if r["found"] else "✗"
            err = f" [{r['error'][:30]}]" if r.get("error") else ""
            print(f"  [{i+1}/{len(test_cases)}] {mark} {tc['id']:12s} "
                  f"({r['num_turns']} turns, {r['total_tokens']} tok, "
                  f"${r['cost_usd']:.4f}, {r['latency_ms']:.0f}ms){err}")
            write_status(args.model, len(test_cases), bare_results, combined_results, phase="bare")

        out_path = RESULTS_DIR / f"cc_bare_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "bare", "results": bare_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("BARE LLM (no Glyphh)", bare_results, args.model)

    # --- Combined (Glyphh + LLM) ---
    if args.mode in ("both", "combined"):
        print("Running combined mode (Glyphh + LLM)...")
        combined_results = []
        for i, tc in enumerate(test_cases):
            r = run_test(tc, args.model, with_glyphh=True)
            combined_results.append(r)
            mark = "✓" if r["found"] else "✗"
            err = f" [{r['error'][:30]}]" if r.get("error") else ""
            print(f"  [{i+1}/{len(test_cases)}] {mark} {tc['id']:12s} "
                  f"({r['num_turns']} turns, {r['total_tokens']} tok, "
                  f"${r['cost_usd']:.4f}, {r['latency_ms']:.0f}ms){err}")
            write_status(args.model, len(test_cases), bare_results, combined_results, phase="combined")

        out_path = RESULTS_DIR / f"cc_combined_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "combined", "results": combined_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("GLYPHH + LLM", combined_results, args.model)

    # --- Comparison ---
    if args.mode == "both" and bare_results and combined_results:
        print("=" * 70)
        print("  HEAD-TO-HEAD: GLYPHH + LLM vs BARE LLM")
        print("=" * 70)

        # Overall
        n = len(test_cases)
        c_found = sum(1 for r in combined_results if r["found"])
        b_found = sum(1 for r in bare_results if r["found"])
        c_cost = sum(r["cost_usd"] for r in combined_results)
        b_cost = sum(r["cost_usd"] for r in bare_results)
        c_tokens = sum(r["total_tokens"] for r in combined_results)
        b_tokens = sum(r["total_tokens"] for r in bare_results)
        c_turns = sum(r["num_turns"] for r in combined_results)
        b_turns = sum(r["num_turns"] for r in bare_results)
        c_latency = sum(r["latency_ms"] for r in combined_results)
        b_latency = sum(r["latency_ms"] for r in bare_results)

        print(f"\n  {'OVERALL':20s} {'Glyphh+LLM':>12s} {'Bare LLM':>12s} {'Delta':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Accuracy':20s} {c_found:>7d}/{n:<4d} {b_found:>7d}/{n:<4d}")
        print(f"  {'Avg tokens':20s} {c_tokens/n:>12.0f} {b_tokens/n:>12.0f} {(1-c_tokens/max(b_tokens,1))*100:>+9.0f}%")
        print(f"  {'Avg turns':20s} {c_turns/n:>12.1f} {b_turns/n:>12.1f} {(1-c_turns/max(b_turns,1))*100:>+9.0f}%")
        print(f"  {'Avg latency':20s} {c_latency/n:>10.0f}ms {b_latency/n:>10.0f}ms {(1-c_latency/max(b_latency,1))*100:>+9.0f}%")
        print(f"  {'Total cost':20s} ${c_cost:>11.4f} ${b_cost:>11.4f} {(1-c_cost/max(b_cost,0.0001))*100:>+9.0f}%")

        # Per type
        types_in_results = sorted(set(r["type"] for r in combined_results))
        for test_type in types_in_results:
            cr = [r for r in combined_results if r["type"] == test_type]
            br = [r for r in bare_results if r["type"] == test_type]
            tn = len(cr)
            cf = sum(1 for r in cr if r["found"])
            bf = sum(1 for r in br if r["found"])
            cc = sum(r["cost_usd"] for r in cr)
            bc = sum(r["cost_usd"] for r in br)
            ct = sum(r["total_tokens"] for r in cr)
            bt = sum(r["total_tokens"] for r in br)

            print(f"\n  {test_type.upper():20s} {'Glyphh+LLM':>12s} {'Bare LLM':>12s} {'Delta':>10s}")
            print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
            print(f"  {'Accuracy':20s} {cf:>7d}/{tn:<4d} {bf:>7d}/{tn:<4d}")
            print(f"  {'Avg tokens':20s} {ct/tn:>12.0f} {bt/tn:>12.0f} {(1-ct/max(bt,1))*100:>+9.0f}%")
            print(f"  {'Cost':20s} ${cc:>11.4f} ${bc:>11.4f} {(1-cc/max(bc,0.0001))*100:>+9.0f}%")

        print()

    # Final status
    write_status(args.model, len(test_cases), bare_results, combined_results, phase="done")

    # Cleanup
    if MCP_CONFIG_FILE.exists():
        MCP_CONFIG_FILE.unlink()


if __name__ == "__main__":
    main()
