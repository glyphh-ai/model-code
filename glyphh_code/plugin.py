"""
Glyphh Code plugin for the Glyphh runtime shell.

Registers as 'code' command via entry points:
    glyphh> code init .
    glyphh> code compile .
    glyphh> code status
    glyphh> code stop
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import click

try:
    from glyphh.cli import theme
except ImportError:
    class _FallbackTheme:
        PRIMARY = "magenta"
        ACCENT = "bright_magenta"
        MUTED = "bright_black"
        SUCCESS = "green"
        WARNING = "yellow"
        ERROR = "red"
        INFO = "cyan"
        TEXT = "white"
        TEXT_DIM = "bright_black"
    theme = _FallbackTheme()

_PACKAGE_DIR = Path(__file__).parent  # glyphh_code package directory
_GLYPHH_DIR = Path.home() / ".glyphh"
_STATE_FILE = _GLYPHH_DIR / "code.json"


def _get_runtime_url() -> str | None:
    """Get the URL of the running dev server."""
    dev_info = _GLYPHH_DIR / "dev.json"
    if not dev_info.exists():
        return None
    try:
        info = json.loads(dev_info.read_text())
        port = info.get("port", 8002)
        return f"http://localhost:{port}"
    except Exception:
        return None


def _compile_repo(repo_path: str, runtime_url: str) -> tuple[int, list[str]]:
    """Compile the repository into the Glyphh index.

    Returns (file_count, job_ids).
    """
    env = {**__import__("os").environ, "PYTHONWARNINGS": "ignore"}
    result = subprocess.run(
        [sys.executable, "-m", "glyphh_code.compile", repo_path,
         "--runtime-url", runtime_url],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        click.secho(f"  Compile error: {result.stderr.strip()[:200]}", fg=theme.ERROR)
        return 0, []

    # Parse file count and job IDs from output
    file_count = 0
    job_ids = []
    for line in result.stdout.strip().split("\n"):
        if "files indexed" in line:
            try:
                file_count = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "Encoded:" in line:
            try:
                file_count = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        if "→ job " in line:
            try:
                job_id = line.split("→ job ")[1].strip()
                if job_id and job_id != "?":
                    job_ids.append(job_id)
            except (IndexError, ValueError):
                pass
    return file_count, job_ids


def _wait_for_jobs(
    job_ids: list[str],
    runtime_url: str,
    org_id: str = "local-dev-org",
    model_id: str = "code",
    timeout: int = 300,
    poll_interval: float = 1.0,
) -> bool:
    """Poll the runtime until all encoding jobs complete.

    Returns True if all jobs completed successfully, False on error/timeout.
    """
    if not job_ids:
        return True

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import requests

    pending = set(job_ids)
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        time.sleep(poll_interval)
        done = set()
        for job_id in pending:
            try:
                r = requests.get(
                    f"{runtime_url}/{org_id}/{model_id}/listener/jobs/{job_id}",
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                status = r.json().get("status", "")
                if status == "completed":
                    done.add(job_id)
                elif status == "error":
                    msg = r.json().get("error", "unknown error")
                    click.secho(f"  Job {job_id[:8]}… failed: {msg}", fg=theme.ERROR)
                    done.add(job_id)
            except Exception:
                pass  # Network blip — retry on next poll
        pending -= done

    if pending:
        click.secho(f"  Warning: {len(pending)} job(s) did not complete in {timeout}s", fg=theme.WARNING)
        return False
    return True


def _deploy_model(runtime_url: str) -> bool:
    """Deploy the code model to the running runtime via API.

    Uses the runtime's package_model to create a .glyphh ZIP,
    then uploads it as a multipart file — same flow as `model deploy`.
    """
    try:
        from glyphh.cli.packaging import package_model
        import httpx

        # Package the model directory into a .glyphh file
        glyphh_file = package_model(_PACKAGE_DIR)

        try:
            with httpx.Client(timeout=60) as client:
                with open(glyphh_file, "rb") as f:
                    r = client.post(
                        f"{runtime_url}/local-dev-org/code/model/deploy",
                        files={"file": (glyphh_file.name, f, "application/octet-stream")},
                    )
            return r.status_code in (200, 201)
        finally:
            # Clean up the temporary .glyphh file
            glyphh_file.unlink(missing_ok=True)
    except Exception as e:
        click.secho(f"  Deploy warning: {e}", fg=theme.WARNING)
        return False


def _configure_claude_code(repo_path: str, mcp_url: str, is_upgrade: bool = False):
    """Configure Claude Code: MCP server, hooks, permissions.

    Does NOT touch the user's CLAUDE.md. Glyphh tool usage is enforced
    via a PreToolUse search gate hook that blocks Grep/Glob/Bash(grep|find)
    until glyphh_search has been called at least once per session.
    """
    repo = Path(repo_path).resolve()
    claude_dir = repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    # 1. Add MCP server to Claude Code
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "--transport", "http", "glyphh", mcp_url],
            capture_output=True,
            text=True,
            cwd=str(repo),
        )
        if result.returncode == 0:
            click.secho("  ✓ MCP server added to Claude Code", fg=theme.SUCCESS)
        else:
            click.secho(f"  ✗ MCP failed: {result.stderr.strip()[:80]}", fg=theme.WARNING)
            click.secho(f"    Run: claude mcp add --transport http glyphh {mcp_url}", fg=theme.TEXT_DIM)
    except FileNotFoundError:
        click.secho("  ✗ Claude Code CLI not found (npm install -g @anthropic-ai/claude-code)", fg=theme.WARNING)
        click.secho(f"    Run: claude mcp add --transport http glyphh {mcp_url}", fg=theme.TEXT_DIM)

    # 2. Migrate: remove Glyphh section from CLAUDE.md if previously injected
    target_claude_md = repo / "CLAUDE.md"
    _GLYPHH_MARKER = "# Glyphh Code Intelligence"
    if target_claude_md.exists():
        existing = target_claude_md.read_text()
        if _GLYPHH_MARKER in existing:
            marker_pos = existing.index(_GLYPHH_MARKER)
            cleaned = existing[:marker_pos].rstrip()
            if cleaned:
                target_claude_md.write_text(cleaned + "\n")
            else:
                target_claude_md.unlink()
            click.secho("  ✓ Migrated Glyphh instructions out of CLAUDE.md", fg=theme.SUCCESS)

    # 3. Add hooks + permissions to .claude/settings.json
    settings_file = claude_dir / "settings.json"

    settings = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            pass

    # MCP tool permissions
    permissions = settings.setdefault("permissions", {})
    allow_list = permissions.setdefault("allow", [])
    if "mcp__glyphh__*" not in allow_list:
        allow_list.append("mcp__glyphh__*")

    # Hooks — on upgrade, refresh paths (package dir may have moved)
    hooks = settings.setdefault("hooks", {})

    # ── SessionStart: reset the search-gate flag ──────────────────────────
    session_hooks = hooks.setdefault("SessionStart", [])
    glyphh_dir = repo / ".glyphh"
    reset_cmd = f"rm -f {glyphh_dir}/.search_used"
    # Remove stale glyphh session hooks, then add current
    session_hooks[:] = [
        h for h in session_hooks
        if ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    session_hooks.append({
        "hooks": [{"type": "command", "command": reset_cmd}],
    })

    # ── PreToolUse: gate Grep/Glob/Bash(grep|find) until glyphh_search called ─
    pre_hooks = hooks.setdefault("PreToolUse", [])
    gate_script = _PACKAGE_DIR / "hooks" / "search-gate.sh"
    gate_cmd = f"bash {gate_script} {glyphh_dir}"
    # Remove stale glyphh gate hooks, then add current
    pre_hooks[:] = [
        h for h in pre_hooks
        if "search-gate" not in h.get("hooks", [{}])[0].get("command", "")
        and ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
        and "enforce-glyphh-search" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    pre_hooks.append({
        "matcher": "Grep|Glob|Bash",
        "hooks": [{"type": "command", "command": gate_cmd}],
    })

    # ── PostToolUse: set search-gate flag after glyphh_search ────────────
    post_hooks = hooks.setdefault("PostToolUse", [])
    flag_cmd = f"mkdir -p {glyphh_dir} && touch {glyphh_dir}/.search_used"
    # Remove stale glyphh flag hooks
    post_hooks[:] = [
        h for h in post_hooks
        if ".search_used" not in h.get("hooks", [{}])[0].get("command", "")
    ]
    post_hooks.append({
        "matcher": "mcp__glyphh__glyphh_search",
        "hooks": [{"type": "command", "command": flag_cmd}],
    })

    # ── PostToolUse: incremental compile after git commits ───────────────
    compile_script = _PACKAGE_DIR / "hooks" / "post-git-compile.sh"
    if compile_script.exists():
        if is_upgrade:
            # Replace existing compile hook with updated path
            post_hooks[:] = [
                h for h in post_hooks
                if "post-git-compile" not in h.get("hooks", [{}])[0].get("command", "")
                and "post-commit-compile" not in h.get("hooks", [{}])[0].get("command", "")
            ]
        else:
            # First init — just clean up legacy hook name
            post_hooks[:] = [
                h for h in post_hooks
                if "post-commit-compile" not in h.get("hooks", [{}])[0].get("command", "")
            ]
        if not any("post-git-compile" in h.get("hooks", [{}])[0].get("command", "") for h in post_hooks):
            post_hooks.append({
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": f"{compile_script} {repo}"}],
            })

    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    click.secho("  ✓ Hooks and permissions configured", fg=theme.SUCCESS)

    # 4. Write .glyphh/manifest.yaml so `model` commands resolve model_id
    glyphh_dir = repo / ".glyphh"
    glyphh_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = glyphh_dir / "manifest.yaml"
    manifest_path.write_text(
        "# Auto-generated by glyphh code init — do not edit\n"
        "model_id: code\n"
        "name: Glyphh Code\n"
    )

    # 5. Add .glyphh/ to .gitignore
    gitignore = repo / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".glyphh/" not in content:
            with open(gitignore, "a") as f:
                f.write("\n# Glyphh local index\n.glyphh/\n")
            click.secho("  ✓ .glyphh/ added to .gitignore", fg=theme.SUCCESS)
    else:
        gitignore.write_text("# Glyphh local index\n.glyphh/\n")
        click.secho("  ✓ .gitignore created", fg=theme.SUCCESS)


def _start_dev_server() -> str | None:
    """Auto-start the dev server if not running. Returns runtime URL."""
    runtime_url = _get_runtime_url()
    if runtime_url:
        return runtime_url

    click.secho("  Starting dev server...", fg=theme.MUTED)
    try:
        from glyphh.cli.commands.dev import handle_dev
        # Start daemon pointing at our package dir (has manifest.yaml)
        handle_dev("start", str(_PACKAGE_DIR))
    except Exception as e:
        click.secho(f"  Could not auto-start: {e}", fg=theme.WARNING)
        click.secho("  Start manually: dev start", fg=theme.ACCENT)
        return None

    # Wait for server to come up
    for _ in range(10):
        time.sleep(1)
        runtime_url = _get_runtime_url()
        if runtime_url:
            return runtime_url

    click.secho("  Server did not start in time.", fg=theme.ERROR)
    return None


def _cmd_init(args: str):
    """code init [path] — setup or upgrade a repository."""
    repo = str(Path(args.strip() or ".").resolve())

    if not Path(repo).is_dir():
        click.secho(f"  Not a directory: {repo}", fg=theme.ERROR)
        return

    # Detect upgrade: state file exists for this repo
    is_upgrade = False
    if _STATE_FILE.exists():
        try:
            prev = json.loads(_STATE_FILE.read_text())
            if prev.get("repo") == repo:
                is_upgrade = True
        except (json.JSONDecodeError, KeyError):
            pass

    runtime_url = _start_dev_server()
    if not runtime_url:
        return

    mode = "upgrade" if is_upgrade else "init"
    click.echo()
    click.secho(f"  Glyphh Code  ·  {mode}", fg=theme.TEXT, bold=True)
    click.secho(f"  {repo}", fg=theme.TEXT_DIM)
    click.echo()

    # Step 1: Deploy model (new weights / encoder on upgrade)
    click.secho("  [1/5] Deploying model...", fg=theme.MUTED)
    _deploy_model(runtime_url)

    # Step 2: Clear existing index (stale data from old weights)
    click.secho("  [2/5] Clearing index...", fg=theme.MUTED)
    try:
        import httpx
        with httpx.Client(timeout=30) as client:
            r = client.delete(f"{runtime_url}/local-dev-org/code/data")
            if r.status_code == 200:
                deleted = r.json().get("glyphs_deleted", 0)
                if deleted:
                    click.secho(f"         {deleted} stale glyphs removed", fg=theme.TEXT_DIM)
    except Exception:
        pass  # Fresh install — nothing to clear

    # Step 3: Full compile
    click.secho("  [3/5] Compiling codebase...", fg=theme.MUTED)
    file_count, job_ids = _compile_repo(repo, runtime_url)
    click.secho(f"         {file_count} files indexed", fg=theme.TEXT_DIM)

    # Step 4: Wait for encoding to complete
    if job_ids:
        click.secho("  [4/5] Encoding...", fg=theme.MUTED)
        _wait_for_jobs(job_ids, runtime_url)
        click.secho("         encoding complete", fg=theme.TEXT_DIM)
    else:
        click.secho("  [4/5] Encoding... skipped (no jobs)", fg=theme.MUTED)

    # Step 5: Configure Claude Code (update paths + instructions on upgrade)
    click.secho("  [5/5] Configuring Claude Code...", fg=theme.MUTED)
    dev_info = json.loads((_GLYPHH_DIR / "dev.json").read_text())
    mcp_url = dev_info.get("mcp_url", f"{runtime_url}/local-dev-org/code/mcp")
    _configure_claude_code(repo, mcp_url, is_upgrade=is_upgrade)

    # Save state
    _STATE_FILE.write_text(json.dumps({
        "repo": repo,
        "runtime_url": runtime_url,
        "mcp_url": mcp_url,
        "file_count": file_count,
    }))

    click.echo()
    dot = click.style("●", fg=theme.SUCCESS)
    label = "upgraded" if is_upgrade else "ready"
    click.echo(f"  {dot} {click.style(label, fg=theme.SUCCESS)}")
    click.echo()
    click.secho(f"  Repo:      {repo}", fg=theme.TEXT_DIM)
    click.secho(f"  Files:     {file_count} indexed", fg=theme.TEXT_DIM)
    click.secho(f"  MCP:       {mcp_url}", fg=theme.ACCENT)
    click.echo()
    click.secho("  Restart Claude Code to activate.", fg=theme.MUTED)
    click.secho("  VS Code: Cmd+Shift+P → 'Claude Code: Restart'", fg=theme.TEXT_DIM)
    click.echo()


def _cmd_compile(args: str):
    """code compile [path] — recompile the index."""
    runtime_url = _start_dev_server()
    if not runtime_url:
        return

    repo = str(Path(args.strip() or ".").resolve())
    click.secho(f"  Compiling: {repo}", fg=theme.TEXT)
    count, job_ids = _compile_repo(repo, runtime_url)
    if job_ids:
        click.secho(f"  Waiting for {len(job_ids)} encoding job(s)...", fg=theme.MUTED)
        _wait_for_jobs(job_ids, runtime_url)
    click.secho(f"  Done: {count} files indexed", fg=theme.SUCCESS)


def _cmd_status(args: str):
    """code status — show current status."""
    if not _STATE_FILE.exists():
        click.secho("  Glyphh Code not initialized.", fg=theme.TEXT_DIM)
        click.secho("  Run: code init /path/to/repo", fg=theme.MUTED)
        return

    state = json.loads(_STATE_FILE.read_text())
    runtime_url = _get_runtime_url()

    click.echo()
    if runtime_url:
        dot = click.style("●", fg=theme.SUCCESS)
        click.echo(f"  {dot} {click.style('running', fg=theme.SUCCESS)}")
    else:
        dot = click.style("●", fg=theme.ERROR)
        click.echo(f"  {dot} {click.style('server not running', fg=theme.ERROR)}")

    click.echo()
    click.secho(f"  Repo:    {state.get('repo', '?')}", fg=theme.TEXT_DIM)
    click.secho(f"  Files:   {state.get('file_count', '?')} indexed", fg=theme.TEXT_DIM)
    click.secho(f"  MCP:     {state.get('mcp_url', '?')}", fg=theme.ACCENT)
    click.echo()
    click.secho("  Model commands work from this repo:", fg=theme.TEXT_DIM)
    click.secho("    model count, model data, model status, model re-encode", fg=theme.MUTED)
    click.echo()


def _cmd_stop(args: str):
    """code stop — stop the dev server."""
    try:
        from glyphh.cli.commands.dev import handle_dev
        handle_dev("stop", args)
    except Exception as e:
        click.secho(f"  Error: {e}", fg=theme.ERROR)


def handle_code(func: str | None, args: str = ""):
    """Route code subcommands from the Glyphh shell."""
    commands = {
        "init": _cmd_init,
        "compile": _cmd_compile,
        "status": _cmd_status,
        "stop": _cmd_stop,
    }

    if func is None:
        _cmd_status(args)
        return

    handler = commands.get(func)
    if handler:
        handler(args)
    else:
        click.secho(f"  Unknown: code {func}", fg=theme.WARNING)
        click.secho("  Available: init, compile, status, stop", fg=theme.TEXT_DIM)


def register():
    """Entry point registration — called by runtime shell plugin discovery."""
    return {
        "handler": handle_code,
        "subcommands": ["init", "compile", "status", "stop"],
    }
