#!/usr/bin/env python3
"""Claude Code benchmark — Glyphh + LLM vs bare LLM.

Runs real Claude Code sessions against a target repo. Measures the actual
tool calls, tokens, and cost Claude Code uses to complete each task.

Test types:
  blast_radius  — "what breaks if I edit X?" (single-call vs multi-grep)
  drift         — semantic drift score for a specific file (Glyphh-only)
  risk          — risk profile for changed files (Glyphh-only)

Two modes:
  combined — Claude Code with Glyphh MCP server + grep/glob/read
  bare     — Claude Code without Glyphh (grep/glob/read only)

Note: drift and risk tests are Glyphh-only — they are skipped in bare mode.

Usage:
    python benchmark/run_claude_benchmark.py                  # both modes
    python benchmark/run_claude_benchmark.py --mode combined  # with Glyphh
    python benchmark/run_claude_benchmark.py --mode bare      # without Glyphh
    python benchmark/run_claude_benchmark.py --limit 5        # subset
    python benchmark/run_claude_benchmark.py --model sonnet   # use Sonnet
    python benchmark/run_claude_benchmark.py --types blast_radius drift risk
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

GLYPHH_GUIDANCE_BLAST = (
    "You have access to Glyphh MCP tools in addition to grep/glob/read.\n"
    "IMPORTANT: For blast radius analysis, ALWAYS call glyphh_related FIRST "
    "with the file path from the query. This returns semantically similar files "
    "that may need coordinated changes — there is no Grep equivalent.\n"
    "After glyphh_related, you may supplement with grep to find additional "
    "importers, but glyphh_related is the primary tool for this task.\n"
    "Always pass detail='minimal' to keep responses lightweight.\n"
)

# Drift: score semantic drift for a file (Glyphh-only).
PROMPT_DRIFT = (
    "TASK: Use the glyphh_drift tool to compute the semantic drift score "
    "for the specified file. Report the drift_score (0.0 to 1.0) and "
    "drift_label (cosmetic, moderate, significant, or architectural).\n"
    "Format your response as:\n"
    "  drift_score: <number>\n"
    "  drift_label: <label>\n"
    "Do NOT ask clarifying questions — just call the tool and report."
)

GLYPHH_GUIDANCE_DRIFT = (
    "You have access to Glyphh MCP tools.\n"
    "IMPORTANT: Call glyphh_drift with the file_path from the query. "
    "This computes how much the file has changed semantically since the "
    "last index build. There is no grep/glob equivalent for this.\n"
)

# Risk: aggregate risk profile for changed files (Glyphh-only).
PROMPT_RISK = (
    "TASK: Use the glyphh_risk tool to compute the risk profile for the "
    "current working tree (or the specified git ref). Report:\n"
    "  risk_label: <cosmetic|moderate|significant|architectural>\n"
    "  max_drift: <number>\n"
    "  mean_drift: <number>\n"
    "  hot_files: <list of files above moderate threshold, or 'none'>\n"
    "Do NOT ask clarifying questions — just call the tool and report."
)

GLYPHH_GUIDANCE_RISK = (
    "You have access to Glyphh MCP tools.\n"
    "IMPORTANT: Call glyphh_risk to get the aggregate risk profile. "
    "This scores all changed files by semantic drift and identifies hot files. "
    "There is no grep/glob equivalent for this.\n"
)


def _get_prompt(test_type: str, with_glyphh: bool) -> str:
    """Return the appropriate system prompt for the test type and mode."""
    base = {
        "blast_radius": PROMPT_BLAST,
        "drift": PROMPT_DRIFT,
        "risk": PROMPT_RISK,
    }.get(test_type, PROMPT_BLAST)

    if with_glyphh:
        guidance = {
            "blast_radius": GLYPHH_GUIDANCE_BLAST,
            "drift": GLYPHH_GUIDANCE_DRIFT,
            "risk": GLYPHH_GUIDANCE_RISK,
        }.get(test_type, GLYPHH_GUIDANCE_BLAST)
        return guidance + "\n" + base
    return base


# Test types that require Glyphh — skipped in bare mode.
GLYPHH_ONLY_TYPES = {"drift", "risk"}


def _get_budget(test_type: str) -> float:
    """Return max budget per test type."""
    return {
        "blast_radius": 0.30,
        "drift": 0.15,
        "risk": 0.15,
    }.get(test_type, 0.30)


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
        "avg_cost": round(sum(r["cost_usd"] for r in results) / n, 4),
        "avg_api_ms": round(sum(r.get("api_ms", r["latency_ms"]) for r in results) / n),
        "avg_turns": round(sum(r["num_turns"] for r in results) / n, 1),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "subagent_spawns": sum(1 for r in results if len(r.get("models_used", [])) > 1),
        "subagent_cost": round(sum(r.get("subagent_cost", 0) for r in results), 4),
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
        b_cost = max(bs.get("avg_cost", 0.0001), 0.0001)
        b_api = max(bs.get("avg_api_ms", 1), 1)
        b_turns = max(bs.get("avg_turns", 1), 0.1)
        status["comparison"] = {
            "cost_savings_pct": round((1 - cs.get("avg_cost", 0) / b_cost) * 100),
            "api_time_savings_pct": round((1 - cs.get("avg_api_ms", 0) / b_api) * 100),
            "turn_savings_pct": round((1 - cs.get("avg_turns", 0) / b_turns) * 100),
            "bare_accuracy": bs.get("accuracy", 0),
            "combined_accuracy": cs.get("accuracy", 0),
            "bare_subagent_spawns": bs.get("subagent_spawns", 0),
            "combined_subagent_spawns": cs.get("subagent_spawns", 0),
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
    """Evaluate whether the result is correct based on test type.

    Matching strategy for blast_radius/semantic (in order):
    1. Exact full-path substring match: "src/fastmcp/server/tasks/handlers.py"
    2. Path with line number suffix: "server/server.py:1121-1145"
    3. Directory + basename: Claude writes `src/.../tasks/` then lists `handlers.py`
       under it — reconstruct and match the full path.

    Drift: checks that drift_score and drift_label appear in output.
    Risk: checks that risk_label and either max_drift or mean_drift appear.
    """
    test_type = test_case["type"]

    if test_type in ("blast_radius", "semantic"):
        expected_files = test_case.get("expected_files", [])
        min_expected = test_case.get("min_expected", 1)
        found_count = 0
        for ef in expected_files:
            # 1. Exact substring
            if ef in result_text:
                found_count += 1
                continue
            # 2. Check without .py extension (handles "base" for "base.py")
            ef_stem = ef.rsplit("/", 1)[-1].replace(".py", "")
            ef_dir = "/".join(ef.split("/")[:-1])  # e.g. "src/fastmcp/server/tasks"
            # 3. Directory + basename: dir mentioned + basename mentioned nearby
            if ef_dir and ef_dir in result_text:
                basename = ef.rsplit("/", 1)[-1]
                if basename in result_text:
                    found_count += 1
                    continue
            # 4. Partial path match: last 2+ segments (e.g. "middleware/error_handling.py")
            parts = ef.split("/")
            for depth in range(2, len(parts)):
                partial = "/".join(parts[-depth:])
                if partial in result_text:
                    found_count += 1
                    break
        return found_count >= min_expected

    if test_type == "drift":
        # Success = Claude called glyphh_drift and reported both score and label.
        text_lower = result_text.lower()
        has_score = "drift_score" in text_lower or "drift score" in text_lower
        has_label = any(label in text_lower for label in
                        ("cosmetic", "moderate", "significant", "architectural"))
        return has_score and has_label

    if test_type == "risk":
        # Success = Claude called glyphh_risk and reported risk profile.
        text_lower = result_text.lower()
        has_risk_label = "risk_label" in text_lower or "risk label" in text_lower or \
            any(label in text_lower for label in
                ("cosmetic", "moderate", "significant", "architectural"))
        has_drift = "drift" in text_lower
        return has_risk_label and has_drift

    return False


# Minimum cache_read tokens for a valid session.  A healthy session loads
# the system prompt + tool definitions into the cache (~40-70K tokens).
# If cache_read is below this threshold the session failed to bootstrap
# (transient network issue, MCP stall, etc.) and should be retried.
MIN_CACHE_READ = 30000
MAX_RETRIES = 2


def _parse_result(test_case: dict, raw: dict, elapsed_ms: float) -> dict:
    """Parse raw claude -p output into a result dict."""
    test_type = test_case["type"]
    if test_type == "drift":
        expected_display = f"drift score for {test_case.get('file_path', '?')}"
    elif test_type == "risk":
        expected_display = f"risk profile{' for ' + test_case['git_ref'] if test_case.get('git_ref') else ''}"
    else:
        expected_display = f"{test_case.get('min_expected', 1)}+ of {len(test_case.get('expected_files', []))} files"

    if "error" in raw:
        return {
            "id": test_case["id"],
            "type": test_type,
            "query": test_case["query"],
            "expected": expected_display,
            "found": False,
            "result_files": [],
            "num_turns": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "latency_ms": round(elapsed_ms, 1),
            "api_ms": 0,
            "models_used": [],
            "subagent_cost": 0,
            "error": raw["error"],
        }

    result_text = raw.get("result", "")
    result_files = extract_files_from_result(result_text)
    found = evaluate_result(test_case, result_text)
    cost = raw.get("total_cost_usd", 0)
    num_turns = raw.get("num_turns", 0)
    duration = raw.get("duration_ms", elapsed_ms)
    api_ms = raw.get("duration_api_ms", duration)

    usage = raw.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    total_tokens = input_tokens + output_tokens + cache_read + cache_create

    # Per-model usage breakdown — detect subagent spawning.
    model_usage = raw.get("modelUsage", {})
    models_used = sorted(model_usage.keys())
    # Subagent cost = total cost of non-primary models (e.g. Haiku Explore agents)
    primary_model_prefix = "claude-sonnet" if "sonnet" in str(models_used) else ""
    if not primary_model_prefix and models_used:
        # Use the most expensive model as primary
        primary_model_prefix = models_used[0].rsplit("-", 1)[0] if models_used else ""
    subagent_cost = sum(
        mu.get("costUSD", 0) for name, mu in model_usage.items()
        if not name.startswith(primary_model_prefix) and primary_model_prefix
    )

    return {
        "id": test_case["id"],
        "type": test_type,
        "query": test_case["query"],
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
        "api_ms": round(api_ms, 1),
        "models_used": models_used,
        "subagent_cost": round(subagent_cost, 6),
        "response_preview": result_text[:300] if result_text else "",
    }


def run_test(
    test_case: dict,
    model: str,
    with_glyphh: bool,
) -> dict:
    """Run a single test case with retry on startup failures.

    A startup failure is detected when cache_read_input_tokens < MIN_CACHE_READ,
    meaning the session never fully loaded the system prompt, tool definitions,
    or MCP tools. This burns budget on retries without producing a real answer.
    """
    for attempt in range(1 + MAX_RETRIES):
        t0 = time.perf_counter()
        raw = run_claude(test_case["query"], model, with_glyphh, test_case["type"])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = _parse_result(test_case, raw, elapsed_ms)

        # Check for startup failure: low cache_read means session didn't bootstrap
        cache_read = result.get("cache_read_tokens", 0)
        if cache_read < MIN_CACHE_READ and attempt < MAX_RETRIES:
            print(f"    ⚠ startup failure (cache_read={cache_read}, "
                  f"cost=${result['cost_usd']:.4f}) — retrying "
                  f"({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(2)  # Brief pause before retry
            continue

        if cache_read < MIN_CACHE_READ:
            result["error"] = result.get("error", "") + f" [startup failure after {MAX_RETRIES} retries]"

        return result

    return result  # unreachable but satisfies type checker


def print_summary(label: str, results: list[dict], model: str):
    """Print summary table with efficiency metrics."""
    total = len(results)
    found = sum(1 for r in results if r["found"])
    accuracy = found / total * 100 if total else 0

    total_cost = sum(r["cost_usd"] for r in results)
    total_subagent = sum(r.get("subagent_cost", 0) for r in results)
    avg_turns = sum(r["num_turns"] for r in results) / total if total else 0
    avg_api_ms = sum(r.get("api_ms", r["latency_ms"]) for r in results) / total if total else 0
    avg_cost = total_cost / total if total else 0

    # Count how many tests spawned subagents (used >1 model)
    subagent_count = sum(1 for r in results if len(r.get("models_used", [])) > 1)

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Model: {model}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:        {found}/{total} ({accuracy:.1f}%)")
    print(f"  Avg cost:        ${avg_cost:.4f}/query")
    print(f"  Avg API time:    {avg_api_ms/1000:.1f}s")
    print(f"  Avg turns:       {avg_turns:.1f}")
    print(f"  Total cost:      ${total_cost:.4f}")
    if subagent_count > 0:
        print(f"  Subagent spawns: {subagent_count}/{total} tests ({subagent_count/total*100:.0f}%)")
        print(f"  Subagent cost:   ${total_subagent:.4f} ({total_subagent/max(total_cost,0.0001)*100:.0f}% of total)")
    print()

    # Per-type breakdown
    print(f"  {'Type':15s} {'Acc':>7s} {'Avg $':>8s} {'Avg API':>8s} {'Turns':>6s} {'Sub$':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")
    for test_type, type_results in sorted(by_type.items()):
        n = len(type_results)
        f = sum(1 for r in type_results if r["found"])
        tc = sum(r["num_turns"] for r in type_results) / n
        cost = sum(r["cost_usd"] for r in type_results) / n
        api = sum(r.get("api_ms", r["latency_ms"]) for r in type_results) / n
        sub = sum(r.get("subagent_cost", 0) for r in type_results)
        print(f"  {test_type:15s} {f:>3d}/{n:<3d} ${cost:>6.4f} {api/1000:>6.1f}s {tc:>5.1f} ${sub:>5.4f}")

    print()

    # Per-test detail for blast_radius (the head-to-head type)
    blast_results = by_type.get("blast_radius", [])
    if blast_results:
        print(f"  Per-test detail (blast_radius):")
        print(f"  {'ID':12s} {'Pass':>4s} {'Cost':>8s} {'API':>6s} {'Turns':>5s} {'Models':s}")
        print(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*6} {'-'*5} {'-'*20}")
        for r in blast_results:
            mark = "✓" if r["found"] else "✗"
            models = "/".join(m.split("-")[1] for m in r.get("models_used", []))
            api_s = r.get("api_ms", r["latency_ms"]) / 1000
            print(f"  {r['id']:12s} {mark:>4s} ${r['cost_usd']:>6.4f} {api_s:>5.1f}s {r['num_turns']:>5d} {models}")
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
        help="Only run specific test types: blast_radius, drift, risk",
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
    # Filter out Glyphh-only test types for bare mode.
    bare_cases = [tc for tc in test_cases if tc["type"] not in GLYPHH_ONLY_TYPES]
    if args.mode in ("both", "bare"):
        print("Running bare LLM mode (no Glyphh)...")
        if len(bare_cases) < len(test_cases):
            skipped = len(test_cases) - len(bare_cases)
            print(f"  (skipping {skipped} Glyphh-only tests: drift, risk)")
        bare_results = []
        for i, tc in enumerate(bare_cases):
            r = run_test(tc, args.model, with_glyphh=False)
            bare_results.append(r)
            mark = "✓" if r["found"] else "✗"
            err = f" [{r['error'][:30]}]" if r.get("error") else ""
            models = "/".join(m.split("-")[1] for m in r.get("models_used", []))
            print(f"  [{i+1}/{len(bare_cases)}] {mark} {tc['id']:12s} "
                  f"({r['num_turns']} turns, ${r['cost_usd']:.4f}, "
                  f"api {r['api_ms']/1000:.0f}s, {models or '?'}){err}")
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
            models = "/".join(m.split("-")[1] for m in r.get("models_used", []))
            print(f"  [{i+1}/{len(test_cases)}] {mark} {tc['id']:12s} "
                  f"({r['num_turns']} turns, ${r['cost_usd']:.4f}, "
                  f"api {r['api_ms']/1000:.0f}s, {models or '?'}){err}")
            write_status(args.model, len(test_cases), bare_results, combined_results, phase="combined")

        out_path = RESULTS_DIR / f"cc_combined_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "mode": "combined", "results": combined_results}, f, indent=2)
        print(f"  Saved: {out_path}")
        print_summary("GLYPHH + LLM", combined_results, args.model)

    # --- Comparison ---
    if args.mode == "both" and bare_results and combined_results:
        comparable_combined = [r for r in combined_results if r["type"] not in GLYPHH_ONLY_TYPES]
        glyphh_only_results = [r for r in combined_results if r["type"] in GLYPHH_ONLY_TYPES]

        print("=" * 70)
        print("  HEAD-TO-HEAD: BLAST RADIUS")
        print("=" * 70)

        n = len(bare_results)

        # Efficiency metrics (the real story)
        c_cost = sum(r["cost_usd"] for r in comparable_combined)
        b_cost = sum(r["cost_usd"] for r in bare_results)
        c_api = sum(r.get("api_ms", r["latency_ms"]) for r in comparable_combined)
        b_api = sum(r.get("api_ms", r["latency_ms"]) for r in bare_results)
        c_turns = sum(r["num_turns"] for r in comparable_combined)
        b_turns = sum(r["num_turns"] for r in bare_results)
        c_found = sum(1 for r in comparable_combined if r["found"])
        b_found = sum(1 for r in bare_results if r["found"])
        c_subs = sum(1 for r in comparable_combined if len(r.get("models_used", [])) > 1)
        b_subs = sum(1 for r in bare_results if len(r.get("models_used", [])) > 1)
        c_subcost = sum(r.get("subagent_cost", 0) for r in comparable_combined)
        b_subcost = sum(r.get("subagent_cost", 0) for r in bare_results)

        print(f"\n  {'Metric':20s} {'Glyphh+LLM':>12s} {'Bare LLM':>12s} {'Savings':>10s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Accuracy':20s} {c_found:>7d}/{n:<4d} {b_found:>7d}/{n:<4d}")
        print(f"  {'Avg cost/query':20s} ${c_cost/n:>10.4f} ${b_cost/n:>10.4f} {(1-c_cost/max(b_cost,0.0001))*100:>+9.0f}%")
        print(f"  {'Avg API time':20s} {c_api/n/1000:>10.1f}s {b_api/n/1000:>10.1f}s {(1-c_api/max(b_api,1))*100:>+9.0f}%")
        print(f"  {'Avg turns':20s} {c_turns/n:>12.1f} {b_turns/n:>12.1f} {(1-c_turns/max(b_turns,1))*100:>+9.0f}%")
        print(f"  {'Subagent spawns':20s} {c_subs:>8d}/{n}   {b_subs:>8d}/{n}")
        if b_subcost > 0 or c_subcost > 0:
            print(f"  {'Subagent cost':20s} ${c_subcost:>10.4f} ${b_subcost:>10.4f}")
        print(f"  {'Total cost':20s} ${c_cost:>10.4f} ${b_cost:>10.4f} {(1-c_cost/max(b_cost,0.0001))*100:>+9.0f}%")

        # Per-test side-by-side
        print(f"\n  Per-test comparison:")
        print(f"  {'ID':12s} {'G.Pass':>6s} {'B.Pass':>6s} {'G.Cost':>8s} {'B.Cost':>8s} {'G.API':>6s} {'B.API':>6s} {'G.Mdl':>8s} {'B.Mdl':>8s}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        for cr, br in zip(comparable_combined, bare_results):
            gm = "✓" if cr["found"] else "✗"
            bm = "✓" if br["found"] else "✗"
            g_models = "/".join(m.split("-")[1] for m in cr.get("models_used", []))
            b_models = "/".join(m.split("-")[1] for m in br.get("models_used", []))
            g_api = cr.get("api_ms", cr["latency_ms"]) / 1000
            b_api = br.get("api_ms", br["latency_ms"]) / 1000
            print(f"  {cr['id']:12s} {gm:>6s} {bm:>6s} ${cr['cost_usd']:>6.4f} ${br['cost_usd']:>6.4f} "
                  f"{g_api:>5.0f}s {b_api:>5.0f}s {g_models:>8s} {b_models:>8s}")

        # Glyphh-only capabilities
        if glyphh_only_results:
            print(f"\n  {'GLYPHH-ONLY CAPABILITIES':s}")
            print(f"  {'-'*50}")
            for test_type in sorted(set(r["type"] for r in glyphh_only_results)):
                gr = [r for r in glyphh_only_results if r["type"] == test_type]
                gn = len(gr)
                gf = sum(1 for r in gr if r["found"])
                gc = sum(r["cost_usd"] for r in gr) / gn
                ga = sum(r.get("api_ms", r["latency_ms"]) for r in gr) / gn / 1000
                print(f"  {test_type.upper():12s}  {gf}/{gn} correct  avg ${gc:.4f}  avg {ga:.1f}s  (no grep equivalent)")

        print()

    # Final status
    write_status(args.model, len(test_cases), bare_results, combined_results, phase="done")

    # Cleanup
    if MCP_CONFIG_FILE.exists():
        MCP_CONFIG_FILE.unlink()


if __name__ == "__main__":
    main()
