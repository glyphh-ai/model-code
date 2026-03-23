#!/usr/bin/env python3
"""Interactive Claude Code benchmark — multi-turn conversations.

Unlike run_claude_benchmark.py (single-prompt mode), this uses Claude Code's
stream-json SDK mode for multi-turn interactive sessions. This captures the
real-world advantage of Glyphh: open-ended questions that force bare Claude
to spawn Explore agents with 30+ tool calls, while Glyphh answers in 1-2.

Measures wall time, tool calls, tokens, turns, and cost per session.

Usage:
    python benchmark/run_interactive_benchmark.py                  # both modes
    python benchmark/run_interactive_benchmark.py --mode combined  # with Glyphh
    python benchmark/run_interactive_benchmark.py --mode bare      # without Glyphh
    python benchmark/run_interactive_benchmark.py --model sonnet
    python benchmark/run_interactive_benchmark.py --limit 5
    python benchmark/run_interactive_benchmark.py --types blast_radius semantic
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
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

MCP_CONFIG = {
    "mcpServers": {
        "glyphh": {
            "type": "http",
            "url": "http://localhost:8002/local-dev-org/code/mcp",
        }
    }
}
MCP_CONFIG_FILE = _BENCHMARK_DIR / ".mcp-interactive.json"
STATUS_FILE = RESULTS_DIR / "interactive_status.json"

# Interactive prompts — open-ended, forcing real exploration
INTERACTIVE_PROMPTS = {
    "blast_radius": [
        # Turn 1: open-ended question (no specific file mentioned)
        "I'm about to make changes to {file}. Before I start, what should I "
        "watch out for? What other parts of the codebase might be affected?",
        # Turn 2: follow-up asking for specifics
        "Can you be more specific about which files would need coordinated "
        "changes? I want to make sure I don't miss anything.",
    ],
    "semantic": [
        # Turn 1: conceptual question
        "{query}",
        # Turn 2: follow-up asking for deeper analysis
        "Can you show me the key files involved and explain how they connect?",
    ],
}

GLYPHH_SYSTEM = (
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


def run_interactive_session(
    turns: list[str],
    model: str,
    with_glyphh: bool,
    budget: float = 0.50,
    timeout: int = 300,
) -> dict:
    """Run a multi-turn interactive Claude Code session via stream-json.

    Returns metrics: wall_time, tool_calls, tokens, turns, cost, response.
    """
    session_id = str(uuid.uuid4())

    cmd = [
        "claude",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--model", model,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--max-budget-usd", str(budget),
        "--verbose",
    ]

    if with_glyphh:
        MCP_CONFIG_FILE.write_text(json.dumps(MCP_CONFIG))
        cmd.extend(["--mcp-config", str(MCP_CONFIG_FILE)])
        cmd.extend(["--append-system-prompt", GLYPHH_SYSTEM])

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    t0 = time.perf_counter()
    tool_calls = []
    all_text = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_create = 0
    cost_usd = 0.0
    turn_count = 0
    errors = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        for turn_idx, turn_text in enumerate(turns):
            # Send user message
            msg = json.dumps({
                "type": "user_message",
                "content": turn_text,
            })
            try:
                proc.stdin.write(msg + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                errors.append("stdin broken pipe")
                break

            turn_count += 1

            # Read streaming events until assistant turn completes
            assistant_done = False
            turn_start = time.perf_counter()

            while not assistant_done:
                # Check timeout
                if time.perf_counter() - t0 > timeout:
                    errors.append("timeout")
                    proc.kill()
                    break

                line = proc.stdout.readline()
                if not line:
                    # Process ended
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # Track tool calls
                if event_type == "tool_use":
                    tool_calls.append({
                        "name": event.get("tool", event.get("name", "unknown")),
                        "turn": turn_idx + 1,
                        "timestamp": time.perf_counter() - t0,
                    })

                # Collect text output
                if event_type == "text" or event_type == "content_block_delta":
                    text = event.get("text", event.get("content", ""))
                    if text:
                        all_text.append(text)

                # Track token usage
                if event_type == "usage" or "usage" in event:
                    usage = event.get("usage", event)
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    total_cache_read += usage.get("cache_read_input_tokens", 0)
                    total_cache_create += usage.get("cache_creation_input_tokens", 0)

                # Track cost
                if "cost_usd" in event:
                    cost_usd = event["cost_usd"]
                elif event_type == "result" and "cost_usd" in event:
                    cost_usd = event["cost_usd"]

                # Detect assistant turn end
                if event_type in ("result", "assistant_message"):
                    assistant_done = True
                    # Extract final usage/cost from result
                    if "usage" in event:
                        u = event["usage"]
                        total_input_tokens = u.get("input_tokens", total_input_tokens)
                        total_output_tokens = u.get("output_tokens", total_output_tokens)
                        total_cache_read = u.get("cache_read_input_tokens", total_cache_read)
                        total_cache_create = u.get("cache_creation_input_tokens", total_cache_create)
                    if "cost_usd" in event:
                        cost_usd = event["cost_usd"]
                    if "total_cost_usd" in event:
                        cost_usd = event["total_cost_usd"]
                    # Collect result text
                    if "result" in event:
                        all_text.append(event["result"])

                # Some formats use message_stop or end_turn
                if event_type in ("message_stop", "end_turn"):
                    assistant_done = True

        # Close stdin to signal we're done
        try:
            proc.stdin.close()
        except Exception:
            pass

        # Wait for process to finish
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Read any remaining stderr
        stderr = proc.stderr.read() if proc.stderr else ""

    except Exception as e:
        errors.append(str(e))
        try:
            proc.kill()
        except Exception:
            pass
        stderr = ""

    wall_time_ms = (time.perf_counter() - t0) * 1000
    total_tokens = total_input_tokens + total_output_tokens + total_cache_read + total_cache_create
    response_text = "\n".join(all_text)

    return {
        "wall_time_ms": round(wall_time_ms),
        "turns": turn_count,
        "tool_calls": len(tool_calls),
        "tool_call_details": tool_calls,
        "total_tokens": total_tokens,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_read_tokens": total_cache_read,
        "cache_create_tokens": total_cache_create,
        "cost_usd": round(cost_usd, 6),
        "response_text": response_text[:2000],
        "errors": errors,
    }


def build_turns(test_case: dict) -> list[str]:
    """Build multi-turn prompts from a test case."""
    test_type = test_case["type"]
    templates = INTERACTIVE_PROMPTS.get(test_type, ["{query}"])

    turns = []
    for template in templates:
        # Substitute variables
        text = template
        if "{file}" in text:
            # Extract file from query (blast radius tests mention a file)
            query = test_case["query"]
            # Try to extract file path from query
            for word in query.split():
                if "/" in word and not word.startswith("http"):
                    text = text.replace("{file}", word.strip("—").strip())
                    break
            else:
                text = text.replace("{file}", "the file mentioned in the query")
        if "{query}" in text:
            text = text.replace("{query}", test_case["query"])
        turns.append(text)

    return turns


def evaluate_result(test_case: dict, response_text: str) -> bool:
    """Evaluate whether the response mentions expected files."""
    expected_files = test_case.get("expected_files", [])
    min_expected = test_case.get("min_expected", 1)
    found_count = sum(1 for ef in expected_files if ef in response_text)
    return found_count >= min_expected


def run_test(test_case: dict, model: str, with_glyphh: bool) -> dict:
    """Run a single interactive test case."""
    turns = build_turns(test_case)
    budget = 0.50  # Higher budget for interactive sessions

    result = run_interactive_session(
        turns=turns,
        model=model,
        with_glyphh=with_glyphh,
        budget=budget,
    )

    found = evaluate_result(test_case, result["response_text"])

    return {
        "id": test_case["id"],
        "type": test_case["type"],
        "query": test_case["query"],
        "expected": f"{test_case.get('min_expected', 1)}+ of {len(test_case.get('expected_files', []))} files",
        "found": found,
        "wall_time_ms": result["wall_time_ms"],
        "turns": result["turns"],
        "tool_calls": result["tool_calls"],
        "total_tokens": result["total_tokens"],
        "cost_usd": result["cost_usd"],
        "errors": result["errors"],
        "response_preview": result["response_text"][:500],
    }


def print_summary(label: str, results: list[dict], model: str):
    """Print summary table."""
    total = len(results)
    if total == 0:
        return
    found = sum(1 for r in results if r["found"])
    accuracy = found / total * 100

    avg_wall = sum(r["wall_time_ms"] for r in results) / total
    avg_tools = sum(r["tool_calls"] for r in results) / total
    avg_tokens = sum(r["total_tokens"] for r in results) / total
    total_cost = sum(r["cost_usd"] for r in results)

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Model: {model} · Interactive mode (multi-turn)")
    print(f"{'=' * 70}")
    print(f"  Accuracy:       {found}/{total} ({accuracy:.1f}%)")
    print(f"  Avg wall time:  {avg_wall/1000:.1f}s")
    print(f"  Avg tool calls: {avg_tools:.1f}")
    print(f"  Avg tokens:     {avg_tokens:.0f}")
    print(f"  Total cost:     ${total_cost:.4f}")
    print()

    for test_type, type_results in sorted(by_type.items()):
        n = len(type_results)
        f = sum(1 for r in type_results if r["found"])
        wt = sum(r["wall_time_ms"] for r in type_results) / n
        tc = sum(r["tool_calls"] for r in type_results) / n
        tk = sum(r["total_tokens"] for r in type_results) / n
        cost = sum(r["cost_usd"] for r in type_results)
        print(f"  {test_type:15s}  {f}/{n} correct  "
              f"avg {wt/1000:.1f}s wall  avg {tc:.1f} tools  "
              f"avg {tk:.0f} tok  ${cost:.4f}")

    print()

    failures = [r for r in results if not r["found"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            err = f"  error: {'; '.join(r['errors'][:2])}" if r.get("errors") else ""
            print(f"    {r['id']}: expected {r['expected']}{err}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Interactive Claude Code benchmark")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--mode", choices=["both", "combined", "bare"], default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="+", default=None,
                        help="Only run specific test types: blast_radius, semantic")
    args = parser.parse_args()

    with open(TEST_CASES_FILE) as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    if args.types:
        test_cases = [t for t in test_cases if t["type"] in args.types]
    if args.limit > 0:
        test_cases = test_cases[:args.limit]

    type_counts: dict[str, int] = {}
    for tc in test_cases:
        type_counts[tc["type"]] = type_counts.get(tc["type"], 0) + 1

    print(f"Interactive Claude Code Benchmark")
    print(f"Model:      {args.model}")
    print(f"Test cases: {len(test_cases)} ({', '.join(f'{v} {k}' for k, v in type_counts.items())})")
    print(f"Mode:       {args.mode}")
    print(f"Repo:       {REPO_ROOT}")
    print(f"Multi-turn: yes (2 turns per test)")
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
            err = f" [{'; '.join(r['errors'][:1])}]" if r.get("errors") else ""
            print(f"  [{i+1}/{len(test_cases)}] {mark} {tc['id']:12s} "
                  f"({r['wall_time_ms']/1000:.1f}s wall, {r['tool_calls']} tools, "
                  f"{r['total_tokens']} tok, ${r['cost_usd']:.4f}){err}")

        out_path = RESULTS_DIR / f"interactive_bare_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "bare", "interactive": True, "results": bare_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("BARE LLM (no Glyphh) — Interactive", bare_results, args.model)

    # --- Combined (Glyphh + LLM) ---
    if args.mode in ("both", "combined"):
        print("Running combined mode (Glyphh + LLM)...")
        combined_results = []
        for i, tc in enumerate(test_cases):
            r = run_test(tc, args.model, with_glyphh=True)
            combined_results.append(r)
            mark = "✓" if r["found"] else "✗"
            err = f" [{'; '.join(r['errors'][:1])}]" if r.get("errors") else ""
            print(f"  [{i+1}/{len(test_cases)}] {mark} {tc['id']:12s} "
                  f"({r['wall_time_ms']/1000:.1f}s wall, {r['tool_calls']} tools, "
                  f"{r['total_tokens']} tok, ${r['cost_usd']:.4f}){err}")

        out_path = RESULTS_DIR / f"interactive_combined_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "combined", "interactive": True, "results": combined_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("GLYPHH + LLM — Interactive", combined_results, args.model)

    # --- Head-to-head ---
    if args.mode == "both" and bare_results and combined_results:
        print("=" * 70)
        print("  HEAD-TO-HEAD: GLYPHH + LLM vs BARE LLM (Interactive)")
        print("=" * 70)

        n = len(test_cases)
        c_found = sum(1 for r in combined_results if r["found"])
        b_found = sum(1 for r in bare_results if r["found"])
        c_wall = sum(r["wall_time_ms"] for r in combined_results)
        b_wall = sum(r["wall_time_ms"] for r in bare_results)
        c_tools = sum(r["tool_calls"] for r in combined_results)
        b_tools = sum(r["tool_calls"] for r in bare_results)
        c_tokens = sum(r["total_tokens"] for r in combined_results)
        b_tokens = sum(r["total_tokens"] for r in bare_results)
        c_cost = sum(r["cost_usd"] for r in combined_results)
        b_cost = sum(r["cost_usd"] for r in bare_results)

        def delta(c_val, b_val):
            if b_val == 0:
                return "n/a"
            return f"{(1 - c_val / b_val) * 100:+.0f}%"

        print(f"\n  {'OVERALL':20s} {'Glyphh+LLM':>12s} {'Bare LLM':>12s} {'Delta':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Accuracy':20s} {c_found:>7d}/{n:<4d} {b_found:>7d}/{n:<4d}")
        print(f"  {'Avg wall time':20s} {c_wall/n/1000:>10.1f}s {b_wall/n/1000:>10.1f}s {delta(c_wall, b_wall):>10s}")
        print(f"  {'Avg tool calls':20s} {c_tools/n:>12.1f} {b_tools/n:>12.1f} {delta(c_tools, b_tools):>10s}")
        print(f"  {'Avg tokens':20s} {c_tokens/n:>12.0f} {b_tokens/n:>12.0f} {delta(c_tokens, b_tokens):>10s}")
        print(f"  {'Total cost':20s} ${c_cost:>11.4f} ${b_cost:>11.4f} {delta(c_cost, b_cost):>10s}")

        # Per type
        types_in_results = sorted(set(r["type"] for r in combined_results))
        for test_type in types_in_results:
            cr = [r for r in combined_results if r["type"] == test_type]
            br = [r for r in bare_results if r["type"] == test_type]
            tn = len(cr)
            cf = sum(1 for r in cr if r["found"])
            bf = sum(1 for r in br if r["found"])
            cw = sum(r["wall_time_ms"] for r in cr)
            bw = sum(r["wall_time_ms"] for r in br)
            ct_tools = sum(r["tool_calls"] for r in cr)
            bt_tools = sum(r["tool_calls"] for r in br)
            cc = sum(r["cost_usd"] for r in cr)
            bc = sum(r["cost_usd"] for r in br)

            print(f"\n  {test_type.upper():20s} {'Glyphh+LLM':>12s} {'Bare LLM':>12s} {'Delta':>10s}")
            print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
            print(f"  {'Accuracy':20s} {cf:>7d}/{tn:<4d} {bf:>7d}/{tn:<4d}")
            print(f"  {'Avg wall time':20s} {cw/tn/1000:>10.1f}s {bw/tn/1000:>10.1f}s {delta(cw, bw):>10s}")
            print(f"  {'Avg tool calls':20s} {ct_tools/tn:>12.1f} {bt_tools/tn:>12.1f} {delta(ct_tools, bt_tools):>10s}")
            print(f"  {'Cost':20s} ${cc:>11.4f} ${bc:>11.4f} {delta(cc, bc):>10s}")

        print()

    # Cleanup
    if MCP_CONFIG_FILE.exists():
        MCP_CONFIG_FILE.unlink()


if __name__ == "__main__":
    main()
