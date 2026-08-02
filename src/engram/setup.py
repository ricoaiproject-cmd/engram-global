"""Implementation of the setup wizard & doctor command.

Design principles:
- Separate "decision/append logic" into pure functions that accept paths as
  arguments (easy to test)
- Keep functions with side effects small and isolated for testability
- All operations are idempotent (safe to run any number of times)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading/writing config.toml (tomllib is read-only, so we hand-roll the writer)
# ---------------------------------------------------------------------------

def read_config_toml(config_file: Path) -> dict[str, Any]:
    """Read config.toml and return a dict. Returns an empty dict if the file is missing or broken."""
    if not config_file.is_file():
        return {}
    try:
        import tomllib
        with config_file.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _toml_scalar(value: Any) -> str:
    """TOML representation of a scalar value. Strings use single quotes (no escaping needed)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"


def write_config_toml(config_file: Path, data: dict[str, Any]) -> None:
    """Write a dict out to config.toml.

    Scalar values are written at the top level; dict values are written as
    [sections] (e.g. room_paths). Section keys are quoted since they may
    contain paths, etc.
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sections: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            sections.append(f"\n[{key}]\n")
            for k, v in value.items():
                sections.append(f"'{k}' = {_toml_scalar(v)}\n")
        else:
            lines.append(f"{key} = {_toml_scalar(value)}\n")
    config_file.write_text("".join(lines + sections), encoding="utf-8")


def merge_config_toml(config_file: Path, updates: dict[str, Any]) -> None:
    """Merge `updates` into the existing config (keeping other keys) and write config.toml back out."""
    existing = read_config_toml(config_file)
    existing.update(updates)
    write_config_toml(config_file, existing)


# ---------------------------------------------------------------------------
# Installing templates
# ---------------------------------------------------------------------------

def copy_templates(dest_dir: Path) -> None:
    """Copy the templates bundled with the package into dest_dir (always overwrites)."""
    import importlib.resources as ir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MEMORY_PROTOCOL.md", "ONBOARDING.md"):
        try:
            ref = ir.files("engram.templates").joinpath(name)
            content = ref.read_bytes()
            (dest_dir / name).write_bytes(content)
        except Exception as e:
            print(f"  Warning: failed to copy template {name}: {e}")


# ---------------------------------------------------------------------------
# Agent registration: Claude Code
# ---------------------------------------------------------------------------

def get_engram_mcp_path() -> Path | None:
    """Return the path to the engram-mcp executable. Returns None if not found."""
    exe_dir = Path(sys.executable).parent
    for name in ("engram-mcp.exe", "engram-mcp"):
        candidate = exe_dir / name
        if candidate.is_file():
            return candidate
    found = shutil.which("engram-mcp")
    if found:
        return Path(found)
    return None


def _claude_cmd() -> str | None:
    """The actual path to the claude CLI.

    npm-based installs end up as claude.cmd; passing bare "claude" to
    subprocess fails with WinError 2 (observed in practice). Always use the
    resolved path from `which` (full path, with extension).
    """
    return shutil.which("claude")


def is_claude_mcp_registered() -> bool:
    """Check whether "engram" appears in `claude mcp list`."""
    cmd = _claude_cmd()
    if cmd is None:
        return False
    try:
        result = subprocess.run(
            [cmd, "mcp", "list"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        return "engram:" in output
    except Exception:
        return False


def _claude_registered_path(cmd: str) -> str | None:
    """Extract engram's registered path from the `claude mcp list` output (None if not registered)."""
    try:
        result = subprocess.run(
            [cmd, "mcp", "list"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        m = re.search(r"engram:\s*(\S+)", output)
        return m.group(1) if m else None
    except Exception:
        return None


def register_claude_mcp(engram_mcp_path: Path) -> tuple[bool, str]:
    """Register the engram MCP with Claude Code. Returns (True, message) on success."""
    cmd = _claude_cmd()
    if cmd is None:
        return False, "claude command not found (Claude Code is not installed)"

    registered = _claude_registered_path(cmd)
    if registered is not None:
        if registered == str(engram_mcp_path):
            return True, "already registered (skipped)"
        # Re-register if the path is stale (e.g. location changed after a reinstall)
        try:
            subprocess.run(
                [cmd, "mcp", "remove", "engram"],
                capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
        except Exception:
            pass

    try:
        result = subprocess.run(
            [cmd, "mcp", "add", "--scope", "user", "engram",
             "--", str(engram_mcp_path)],
            capture_output=True, text=True, timeout=90,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return True, "registration complete"
        else:
            return False, f"registration failed (exit {result.returncode}): {result.stderr.strip()}"
    except Exception as e:
        return False, f"registration failed: {e}"


def get_engram_cli_path() -> Path | None:
    """Return the path to the engram CLI itself (used for hook registration). None if not found."""
    exe_dir = Path(sys.executable).parent
    for name in ("engram.exe", "engram"):
        candidate = exe_dir / name
        if candidate.is_file():
            return candidate
    found = shutil.which("engram")
    if found:
        return Path(found)
    return None


#: Hooks registered with Claude Code: (event name, engram hook argument, timeout in seconds)
_CLAUDE_HOOK_EVENTS: list[tuple[str, str, int]] = [
    ("SessionEnd", "session-end", 180),      # Longer timeout: involves model load + summarization
    ("UserPromptSubmit", "user-prompt", 15),  # Lightweight path; don't make the prompt wait
]


def _is_engram_hook_cmd(command: str) -> bool:
    """Determine whether an existing hook command belongs to engram (used for updates/idempotency)."""
    return "engram" in command and " hook " in f"{command} "


def register_claude_hooks(
    settings_path: Path,
    engram_cli: Path,
) -> tuple[bool, str]:
    """Register the auto-encoding / spontaneous-recall hooks in ~/.claude/settings.json. Idempotent.

    - Skips if the same command is already registered
    - Updates the command if it's an engram hook but the path is stale
    - Leaves non-engram hooks untouched
    - Does not modify broken JSON (avoids destructive changes)
    """
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        if settings_path.is_file():
            raw = settings_path.read_text(encoding="utf-8")
            if raw.strip() == "":
                data: dict = {}
            else:
                try:
                    data = json.loads(raw)
                except Exception:
                    return False, "Existing settings.json could not be parsed, so no changes were made (please fix it manually)"
        else:
            data = {}

        hooks = data.setdefault("hooks", {})
        changed = []
        for event, hook_arg, timeout in _CLAUDE_HOOK_EVENTS:
            command = f'"{engram_cli}" hook {hook_arg}'
            entries = hooks.setdefault(event, [])
            found = False
            for entry in entries:
                for h in entry.get("hooks", []):
                    if _is_engram_hook_cmd(h.get("command", "")):
                        found = True
                        if h.get("command") != command:
                            h["command"] = command
                            changed.append(f"{event} (path updated)")
                        h.setdefault("timeout", timeout)
            if not found:
                entries.append({
                    "hooks": [{
                        "type": "command",
                        "command": command,
                        "timeout": timeout,
                    }]
                })
                changed.append(event)

        if not changed:
            return True, "already registered (skipped)"
        settings_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return True, f"registration complete ({', '.join(changed)})"
    except Exception as e:
        return False, f"registration failed: {e}"


def update_claude_md(claude_md_path: Path, protocol_path: Path) -> tuple[bool, str]:
    """Append the memory protocol's @import to ~/.claude/CLAUDE.md. Idempotent."""
    try:
        claude_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if claude_md_path.is_file():
            existing = claude_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "already appended (skipped)"
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = f"\n# Memory Protocol (engram)\n\n@{abs_path}\n"
        with claude_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "append complete"
    except Exception as e:
        return False, f"append failed: {e}"


# ---------------------------------------------------------------------------
# Agent registration: Codex
# ---------------------------------------------------------------------------

# Codex's default MCP initial-connection timeout (30s) isn't always enough
# for engram to start up (loading the ~90MB embedding model + checking the
# memory folder) (connection failures observed in practice). We avoid this
# by explicitly setting a longer startup timeout at registration time.
_CODEX_STARTUP_TIMEOUT_SEC = "120.0"


def register_codex(
    codex_config_path: Path,
    engram_mcp_path: Path,
) -> tuple[bool, str]:
    """Append the engram MCP block to ~/.codex/config.toml. Idempotent.

    If the registered block is missing startup_timeout_sec, it gets appended
    (so users who registered with an older version can fix it just by
    re-running setup).
    """
    try:
        import re

        codex_config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if codex_config_path.is_file():
            existing = codex_config_path.read_text(encoding="utf-8")
        mcp_path_str = str(engram_mcp_path).replace("\\", "/")
        if "[mcp_servers.engram]" in existing:
            # Even if already registered, update it if the path is stale (e.g.
            # location changed after a reinstall). Leaving a broken path in
            # place makes the connection fail (observed in practice).
            # When Codex itself rewrites config.toml, quotes can change to
            # double quotes, so accept either quote style.
            pattern = (
                r"(\[mcp_servers\.engram\]\s*\r?\ncommand = )(['\"])([^'\"\r\n]*)\2"
            )
            m = re.search(pattern, existing)
            if not m:
                return True, "already appended (manual edit detected, left unchanged)"
            updated = existing
            actions = []
            if m.group(3) != mcp_path_str:
                updated = re.sub(
                    pattern,
                    lambda mm: f"{mm.group(1)}{mm.group(2)}{mcp_path_str}{mm.group(2)}",
                    updated,
                )
                actions.append("path updated")
            # If the engram block itself (up to the next section heading) has
            # no startup timeout, append it right after the command line
            block = re.search(
                r"\[mcp_servers\.engram\]\r?\n(?:(?!\[).*\r?\n?)*", updated
            )
            if block and "startup_timeout_sec" not in block.group(0):
                updated = re.sub(
                    r"(\[mcp_servers\.engram\]\s*\r?\ncommand = ['\"][^'\"\r\n]*['\"]\r?\n)",
                    lambda mm: (
                        f"{mm.group(1)}"
                        f"startup_timeout_sec = {_CODEX_STARTUP_TIMEOUT_SEC}\n"
                    ),
                    updated,
                )
                actions.append("added startup timeout")
            if not actions:
                return True, "already appended (skipped)"
            codex_config_path.write_text(updated, encoding="utf-8")
            return True, ", ".join(actions)
        addition = (
            f"\n[mcp_servers.engram]\n"
            f"command = '{mcp_path_str}'\n"
            f"startup_timeout_sec = {_CODEX_STARTUP_TIMEOUT_SEC}\n"
        )
        with codex_config_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "append complete"
    except Exception as e:
        return False, f"append failed: {e}"


def update_agents_md(
    agents_md_path: Path,
    protocol_path: Path,
) -> tuple[bool, str]:
    """Append the full memory protocol text to ~/.codex/AGENTS.md. Idempotent."""
    try:
        agents_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if agents_md_path.is_file():
            existing = agents_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "already appended (skipped)"
        protocol_text = ""
        if protocol_path.is_file():
            protocol_text = protocol_path.read_text(encoding="utf-8")
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = (
            f"\n{protocol_text}\n"
            f"> Source of truth: {abs_path}\n"
        )
        with agents_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "append complete"
    except Exception as e:
        return False, f"append failed: {e}"


# ---------------------------------------------------------------------------
# Agent registration: Gemini / Antigravity
# ---------------------------------------------------------------------------

def register_gemini_mcp(
    mcp_config_path: Path,
    engram_mcp_path: Path,
) -> tuple[bool, str]:
    """Add an engram entry to ~/.gemini/config/mcp_config.json. Idempotent."""
    try:
        mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
        if mcp_config_path.is_file():
            raw = mcp_config_path.read_text(encoding="utf-8")
            if raw.strip() == "":
                # An empty file can be treated as "no config" (observed in practice)
                data = {"mcpServers": {}}
            else:
                try:
                    data = json.loads(raw)
                except Exception:
                    # Overwriting a broken existing config with an empty one
                    # would wipe out other servers' registrations. Stop
                    # without touching it and ask for manual repair (avoid
                    # destructive changes)
                    return False, "Existing mcp_config.json could not be parsed, so no changes were made (please fix it manually)"
        else:
            data = {"mcpServers": {}}

        if "mcpServers" not in data:
            data["mcpServers"] = {}

        mcp_path_str = str(engram_mcp_path).replace("\\", "/")
        if "engram" in data["mcpServers"]:
            current = data["mcpServers"]["engram"].get("command", "")
            if current == mcp_path_str:
                return True, "already registered (skipped)"
            # Update if the path is stale (don't leave a broken path in place)
            data["mcpServers"]["engram"]["command"] = mcp_path_str
            mcp_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True, "path updated"

        data["mcpServers"]["engram"] = {"command": mcp_path_str}
        mcp_config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True, "registration complete"
    except Exception as e:
        return False, f"registration failed: {e}"


def update_gemini_md(
    gemini_md_path: Path,
    protocol_path: Path,
) -> tuple[bool, str]:
    """Append a summary of the memory protocol to ~/.gemini/GEMINI.md. Idempotent."""
    try:
        gemini_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if gemini_md_path.is_file():
            existing = gemini_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "already appended (skipped)"
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = (
            f"\n# Memory Protocol (engram)\n\n"
            f"Always call `recall` before starting a task, and `reinforce` "
            f"when the task is done. Use `remember` for important findings, "
            f"and `correct` (not `forget`) for mistakes. "
            f"See {abs_path} for details.\n"
        )
        with gemini_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "append complete"
    except Exception as e:
        return False, f"append failed: {e}"


# ---------------------------------------------------------------------------
# Agent selection utilities
# ---------------------------------------------------------------------------

#: internal key -> display name
_AGENT_DISPLAY: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex",
    "gemini": "Antigravity (Gemini)",
}

#: input alias -> internal key (lowercase-normalized)
_AGENT_ALIASES: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "gemini": "gemini",
    "antigravity": "gemini",
}

_VALID_AGENT_KEYS = frozenset(_AGENT_DISPLAY.keys())


def parse_agents(value: str) -> set[str]:
    """Convert a comma-separated list of agent names into a set of internal keys.

    Alias: "antigravity" -> "gemini" (case-insensitive).
    Raises ValueError if an invalid name is included (the message lists valid names).
    """
    valid_aliases = sorted(_AGENT_ALIASES.keys())
    result: set[str] = set()
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token not in _AGENT_ALIASES:
            raise ValueError(
                f"Unknown agent name: '{raw.strip()}'\n"
                f"Valid names: {', '.join(valid_aliases)}"
            )
        result.add(_AGENT_ALIASES[token])
    return result


def _detect_agents(
    codex_dir: Path | None = None,
    gemini_dir: Path | None = None,
) -> list[str]:
    """Return the list of internal keys for agents detected in the current environment."""
    _codex_dir = codex_dir if codex_dir is not None else Path.home() / ".codex"
    _gemini_dir = gemini_dir if gemini_dir is not None else Path.home() / ".gemini"
    detected: list[str] = []
    if shutil.which("claude") is not None:
        detected.append("claude")
    if _codex_dir.is_dir():
        detected.append("codex")
    if _gemini_dir.is_dir():
        detected.append("gemini")
    return detected


# ---------------------------------------------------------------------------
# Setup wizard main body
# ---------------------------------------------------------------------------

def setup_main(
    memories_dir: Path | None = None,
    non_interactive: bool = False,
    *,
    agents: set[str] | None = None,
    # Path injection (for testing/customization)
    engram_home: Path | None = None,
    config_file: Path | None = None,
    claude_md_path: Path | None = None,
    codex_dir: Path | None = None,
    gemini_dir: Path | None = None,
) -> None:
    """Main routine for the setup wizard. Idempotent.

    agents: set of internal keys for the agents to register
            ("claude"/"codex"/"gemini"). None = all detected agents,
            as before.
    """
    from .config import config_path as _config_path
    from .config import _engram_home as _get_engram_home
    from .config import get_settings
    from .store import MarkdownStore

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    home = engram_home if engram_home is not None else _get_engram_home()
    cfg_path = config_file if config_file is not None else _config_path()

    results: list[tuple[str, bool, str]] = []

    print("=" * 60)
    print("engram Setup Wizard")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Decide memories_dir and write it to config.toml
    # ------------------------------------------------------------------
    print("[1/6] Creating the config file")

    existing_cfg = read_config_toml(cfg_path)

    if memories_dir is not None:
        chosen_dir = Path(memories_dir)
        print(f"  memories_dir: {chosen_dir} (specified via argument)")
    elif "memories_dir" in existing_cfg:
        chosen_dir = Path(existing_cfg["memories_dir"])
        print(f"  Respecting the existing config file: memories_dir={chosen_dir}")
        print(f"  Config file: {cfg_path}")
        print()
        results.append(("Config file", True, f"Using existing: {cfg_path}"))
    elif non_interactive:
        chosen_dir = home / "memories"
        print(f"  memories_dir: {chosen_dir} (default)")
    else:
        default = home / "memories"
        print()
        print("  Please specify the directory where memories will be stored.")
        print("  Pointing this at a Google Drive or OneDrive sync folder backs up")
        print("  your memories automatically. The search DB always stays local,")
        print("  so this is safe.")
        print()
        answer = input(f"  memories_dir [{default}]: ").strip()
        chosen_dir = Path(answer) if answer else default

    if "memories_dir" not in existing_cfg or memories_dir is not None:
        chosen_dir = chosen_dir.expanduser().resolve()
        merge_config_toml(cfg_path, {"memories_dir": str(chosen_dir)})
        print(f"  Config file created: {cfg_path}")
        print(f"  memories_dir: {chosen_dir}")
        results.append(("Config file creation", True, str(cfg_path)))

    # Explicitly set the spontaneous-recall mode (default: shadow = log-only, observe first)
    if "surface_mode" not in read_config_toml(cfg_path):
        merge_config_toml(cfg_path, {"surface_mode": "shadow"})
        print("  surface_mode: shadow (spontaneous recall starts in log-only observation mode)")

    chosen_dir = Path(existing_cfg.get("memories_dir", chosen_dir)).expanduser().resolve() \
        if "memories_dir" in existing_cfg and memories_dir is None else chosen_dir.expanduser().resolve()

    # ------------------------------------------------------------------
    # Step 2: Initialize the memory folder
    # ------------------------------------------------------------------
    if "memories_dir" not in existing_cfg or memories_dir is not None:
        print()
    print("[2/6] Initializing the memory folder")
    try:
        MarkdownStore(chosen_dir)
        print(f"  Memory folder initialized: {chosen_dir}")
        results.append(("Memory folder initialization", True, str(chosen_dir)))
    except Exception as e:
        results.append(("Memory folder initialization", False, str(e)))
        print(f"  Error: {e}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Install templates
    # ------------------------------------------------------------------
    print("[3/6] Copying templates")
    try:
        copy_templates(home)
        print(f"  MEMORY_PROTOCOL.md -> {home / 'MEMORY_PROTOCOL.md'}")
        print(f"  ONBOARDING.md      -> {home / 'ONBOARDING.md'}")
        results.append(("Template installation", True, "complete"))
    except Exception as e:
        results.append(("Template installation", False, str(e)))
        print(f"  Error: {e}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Fetch the embedding model
    # ------------------------------------------------------------------
    print("[4/6] Fetching the embedding model")
    print("  Downloads roughly 90MB on first run (skipped if already cached)...")
    embedder = None
    try:
        from .embedder import RuriEmbedder
        embedder = RuriEmbedder()
        _ = embedder.dim  # Load on the main thread (known issue: much slower on a worker thread)
        results.append(("Embedding model fetch", True, "complete"))
        print("  Embedding model is ready")
    except Exception as e:
        results.append(("Embedding model fetch", False, str(e)))
        print(f"  Warning: failed to fetch the model ({e})")
        print("  This will be retried the first time the server starts.")
    print()

    # ------------------------------------------------------------------
    # Step 5: Register agents
    # ------------------------------------------------------------------
    print("[5/6] Registering with agents")

    engram_mcp = get_engram_mcp_path()
    if engram_mcp is None:
        print("  Warning: engram-mcp not found. Please check your installation.")
        results.append(("engram-mcp lookup", False, "not found"))
    else:
        print(f"  engram-mcp: {engram_mcp}")

    # --- Agent selection ---
    _codex_dir = codex_dir if codex_dir is not None else Path.home() / ".codex"
    _gemini_dir = gemini_dir if gemini_dir is not None else Path.home() / ".gemini"

    # If --agents is given explicitly, use it as-is regardless of detection
    # Not given + interactive -> show a selection prompt when 2+ agents are detected
    # Not given + non-interactive -> all detected agents (as before)
    if agents is not None:
        # Explicit via --agents: use the given set as-is
        selected_agents: set[str] = agents
        detected_agents: list[str] = _detect_agents(_codex_dir, _gemini_dir)
    else:
        detected_agents = _detect_agents(_codex_dir, _gemini_dir)
        if non_interactive or len(detected_agents) <= 1:
            # Non-interactive, or 1 or fewer choices -> all
            selected_agents = set(detected_agents)
        else:
            # Interactive mode: present choices
            print()
            print("  Detected agents:")
            for idx, key in enumerate(detected_agents, 1):
                print(f"    [{idx}] {_AGENT_DISPLAY[key]}")
            print("  Choose which to register (Enter=all / comma-separated numbers, e.g. 1,3):")
            try:
                answer = input("  > ").strip()
            except EOFError:
                answer = ""
            if not answer:
                selected_agents = set(detected_agents)
            else:
                chosen: set[str] = set()
                valid = True
                for part in answer.split(","):
                    part = part.strip()
                    if not part.isdigit():
                        print(f"  Invalid input: '{part}' — registering all agents")
                        valid = False
                        break
                    n = int(part)
                    if 1 <= n <= len(detected_agents):
                        chosen.add(detected_agents[n - 1])
                    else:
                        print(f"  Number out of range: {n} — registering all agents")
                        valid = False
                        break
                selected_agents = chosen if valid else set(detected_agents)
            print()

    # --- Claude Code ---
    _claude_md = claude_md_path if claude_md_path is not None else Path.home() / ".claude" / "CLAUDE.md"
    if "claude" in selected_agents:
        if shutil.which("claude") is not None and engram_mcp is not None:
            ok, msg = register_claude_mcp(engram_mcp)
            results.append(("Claude Code MCP registration", ok, msg))
            print(f"  Claude Code MCP registration: {msg}")

            ok2, msg2 = update_claude_md(_claude_md, home / "MEMORY_PROTOCOL.md")
            results.append(("CLAUDE.md update", ok2, msg2))
            print(f"  CLAUDE.md update: {msg2}")

            # Hook registration (auto-encoding + spontaneous recall)
            engram_cli = get_engram_cli_path()
            if engram_cli is not None:
                ok3, msg3 = register_claude_hooks(
                    _claude_md.parent / "settings.json", engram_cli
                )
                results.append(("Claude Code hook registration", ok3, msg3))
                print(f"  Claude Code hook registration: {msg3}")
                print("    - Automatic memory on session end (SessionEnd)")
                print("    - Spontaneous recall of related memories (UserPromptSubmit, shadow mode initially)")
            else:
                results.append(("Claude Code hook registration", False, "engram CLI not found"))
                print("  Claude Code hook registration: engram CLI not found")
        else:
            # Explicitly specified via --agents but not detected
            print("  Claude Code: not detected (if installed, re-run `engram setup` later)")
    else:
        # Determine and show why it's not in selected_agents
        if "claude" in detected_agents:
            # Detected but not selected (excluded via interactive prompt / --agents)
            print(f"  {_AGENT_DISPLAY['claude']}: not selected (skipped)")
        else:
            # Not detected at all (agents=None targets all detected, but none exist)
            print("  Claude Code: not detected (if installed, re-run `engram setup` later)")

    # --- Codex ---
    if "codex" in selected_agents:
        if _codex_dir.is_dir() and engram_mcp is not None:
            ok, msg = register_codex(_codex_dir / "config.toml", engram_mcp)
            results.append(("Codex config.toml update", ok, msg))
            print(f"  Codex config.toml update: {msg}")

            ok2, msg2 = update_agents_md(_codex_dir / "AGENTS.md", home / "MEMORY_PROTOCOL.md")
            results.append(("Codex AGENTS.md update", ok2, msg2))
            print(f"  Codex AGENTS.md update: {msg2}")
        else:
            print("  Codex: not detected (if installed, re-run `engram setup` later)")
    else:
        if "codex" in detected_agents:
            print(f"  {_AGENT_DISPLAY['codex']}: not selected (skipped)")
        else:
            print("  Codex: not detected (if installed, re-run `engram setup` later)")

    # --- Gemini / Antigravity ---
    if "gemini" in selected_agents:
        if _gemini_dir.is_dir() and engram_mcp is not None:
            ok, msg = register_gemini_mcp(
                _gemini_dir / "config" / "mcp_config.json",
                engram_mcp,
            )
            results.append(("Gemini mcp_config.json update", ok, msg))
            print(f"  Gemini mcp_config.json update: {msg}")

            ok2, msg2 = update_gemini_md(_gemini_dir / "GEMINI.md", home / "MEMORY_PROTOCOL.md")
            results.append(("Gemini GEMINI.md update", ok2, msg2))
            print(f"  Gemini GEMINI.md update: {msg2}")
        else:
            print("  Antigravity/Gemini: not detected (if installed, re-run `engram setup` later)")
    else:
        if "gemini" in detected_agents:
            print(f"  {_AGENT_DISPLAY['gemini']}: not selected (skipped)")
        else:
            print("  Antigravity/Gemini: not detected (if installed, re-run `engram setup` later)")

    print()

    # ------------------------------------------------------------------
    # Step 6: Health check + stats
    # ------------------------------------------------------------------
    print("[6/6] Health check")
    if embedder is None:
        results.append(("Health check", False, "skipped, model not fetched"))
        print("  Skipped because the model wasn't fetched (will be retried on first server start)")
    else:
        try:
            from .config import get_settings
            from .engine import build_engine
            settings = get_settings()
            # Reuse the real embedder already loaded in Step 4 (a test
            # FakeEmbedder would have a dimension mismatch with the existing
            # DB and fail to open)
            engine = build_engine(settings, embedder=embedder)
            engine.recall("setup check", mode="fast", limit=1,
                          record_hits=False)
            stats = engine.stats()
            engine.db.close()
            n = stats.get("total_memories", 0)
            print(f"  Health check [OK]: {n} memories")
            results.append(("Health check", True, f"{n} memories"))
        except Exception as e:
            results.append(("Health check", False, str(e)))
            print(f"  Warning: {e}")
    print()

    # ------------------------------------------------------------------
    # Completion guidance
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Setup Results")
    print("=" * 60)
    max_step = max((len(s) for s, _, _ in results), default=20)
    for step, ok, msg in results:
        mark = "[OK]" if ok else "[NG]"
        print(f"  {mark} {step:<{max_step}}  {msg}")

    onboarding_path = home / "ONBOARDING.md"
    print()
    print("Next steps:")
    print("  1. Restart your agent (start a new session)")
    print(f"  2. It helps to go through the ONBOARDING.md interview first")
    print(f"     Ask your agent: \"Read {onboarding_path} and interview me\"")
    print()


# ---------------------------------------------------------------------------
# doctor command main body
# ---------------------------------------------------------------------------

def check_embed_backend(settings) -> tuple[str, str]:
    """Diagnose the embedding runtime backend. Returns (status, detail) for a doctor row.

    Returns [OK] with the dim/parity if the ONNX model has been generated,
    or [--] with instructions to run export-onnx if not. If
    embed_backend=torch, reports that instead.
    """
    backend = getattr(settings, "embed_backend", "auto")
    onnx_dir = settings.onnx_model_dir
    meta_path = onnx_dir / "meta.json"

    if backend == "torch":
        return "[--]", "torch is forced (startup is heavy; switch back to auto to use ONNX)"

    if meta_path.is_file():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            parity = meta.get("parity", {}).get("min_cosine")
            parity_str = f", parity={parity:.6f}" if parity is not None else ""
            return "[OK]", (
                f"ONNX (dim={meta.get('dim')}{parity_str})  {onnx_dir}"
            )
        except Exception:
            return "[NG]", f"meta.json is corrupted: {meta_path}"
    return "[--]", (
        "ONNX not generated (falls back to torch, which makes startup heavy). "
        "Please run `engram export-onnx`"
    )


def find_install_remnants(purelib: Path | None = None) -> list[str]:
    """Look for leftover remnants in site-packages from a failed pip reinstall.

    During an update, pip temporarily renames the package to something like
    "~ngram". If the process dies mid-rename (observed case: a running
    engram process had the file locked), only the remnant is left behind and
    `import engram` breaks. Fix: stop the engram process -> delete the
    remnant -> reinstall.
    """
    if purelib is None:
        import sysconfig
        purelib = Path(sysconfig.get_paths()["purelib"])
    if not purelib.is_dir():
        return []
    return sorted(p.name for p in purelib.glob("~*gram*"))


def check_fts5() -> tuple[str, str]:
    """Diagnose whether FTS5 (trigram tokenizer) is available. Returns (status, detail).

    Verifies by actually creating an FTS5 virtual table with
    `tokenize='trigram'` in an in-memory DB (requires SQLite>=3.34). On
    failure, notes that db.py's keyword_search will silently swallow the
    OperationalError and degrade to vector-search-only (this is a silent
    degradation, so we surface it here).
    """
    import sqlite3

    sqlite_version = sqlite3.sqlite_version
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE t USING fts5"
                "(content, tokenize='trigram')"
            )
        finally:
            conn.close()
        return "[OK]", f"sqlite3 {sqlite_version}"
    except sqlite3.OperationalError:
        return "[NG]", (
            f"sqlite3 {sqlite_version} (trigram not supported). "
            "keyword_search will silently degrade to vector-search-only"
        )


def summarize_perf(perf_log_path: Path, max_lines: int = 500) -> tuple[str, str]:
    """Build a recall p50 / preload summary from the recent entries in perf_log.jsonl.

    Returns (status, detail). If the file is missing, returns [--] with a
    note that nothing has been recorded. Skips broken lines (not JSON, or
    missing keys).
    """
    if not perf_log_path.is_file():
        return "[--]", "not recorded yet (created after the first tool call)"

    try:
        with perf_log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return "[NG]", f"read error: {perf_log_path}"

    tail = lines[-max_lines:]
    recall_ms: list[float] = []
    last_preload_ms: float | None = None
    n_parsed = 0

    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            kind = entry["kind"]
            ms = float(entry["ms"])
        except Exception:
            continue
        n_parsed += 1
        if kind == "tool" and entry.get("name") == "recall":
            recall_ms.append(ms)
        elif kind == "preload":
            last_preload_ms = ms

    if n_parsed == 0:
        return "[--]", "no valid records"

    detail_parts = []
    if recall_ms:
        recall_ms.sort()
        mid = len(recall_ms) // 2
        if len(recall_ms) % 2 == 0:
            p50 = (recall_ms[mid - 1] + recall_ms[mid]) / 2
        else:
            p50 = recall_ms[mid]
        detail_parts.append(f"recall p50 {p50:.0f}ms")
    if last_preload_ms is not None:
        detail_parts.append(f"preload {last_preload_ms:.0f}ms")

    if not detail_parts:
        return "[--]", f"no recall/preload records (last {n_parsed} entries)"

    detail = " / ".join(detail_parts) + f" (last {n_parsed} entries)"
    return "[OK]", detail


def doctor_main(
    *,
    engram_home: Path | None = None,
    config_file: Path | None = None,
) -> None:
    """Display environment diagnostics as a table (no engine construction, no model load)."""
    import platform
    from .config import config_path as _config_path
    from .config import _engram_home as _get_engram_home
    from .config import get_settings

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    home = engram_home if engram_home is not None else _get_engram_home()
    cfg_file = config_file if config_file is not None else _config_path()

    print("=" * 60)
    print("engram doctor -- Environment Diagnostics")
    print("=" * 60)
    print()

    W = 38  # label column width

    def row(label: str, status: str, detail: str = "") -> None:
        detail_str = f"  {detail}" if detail else ""
        print(f"  {label:<{W}} {status}{detail_str}")

    # Python version
    py_ver = platform.python_version()
    major, minor, *_ = py_ver.split(".")
    py_ok = int(major) >= 3 and int(minor) >= 12
    row("Python version", "[OK]" if py_ok else "[NG]", py_ver)

    # SQLite extension-loading support (required for sqlite-vec; not supported by macOS's stock Python)
    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(":memory:")
    ext_ok = hasattr(_conn, "enable_load_extension")
    _conn.close()
    row("SQLite extension-loading support", "[OK]" if ext_ok else "[NG]",
        "" if ext_ok else "Please reinstall with a uv-managed Python (see README)")

    # FTS5 (trigram) support. Without it, keyword_search silently degrades to vector-search-only
    fts5_status, fts5_detail = check_fts5()
    row("FTS5 (trigram) support", fts5_status, fts5_detail)

    # Leftover remnants from a failed pip reinstall (e.g. ~ngram) break the import itself
    remnants = find_install_remnants()
    row("Install health", "[OK]" if not remnants else "[NG]",
        "" if not remnants else (
            f"Remnants found in site-packages: {', '.join(remnants)} "
            "(stop the engram process, delete the remnants, and reinstall)"
        ))
    print()

    # config.toml
    cfg_exists = cfg_file.is_file()
    cfg_parseable = False
    cfg_data: dict = {}
    if cfg_exists:
        cfg_data = read_config_toml(cfg_file)
        cfg_parseable = bool(cfg_data) or cfg_file.stat().st_size < 10  # also allow an empty file
        try:
            import tomllib
            with cfg_file.open("rb") as f:
                tomllib.load(f)
            cfg_parseable = True
        except Exception:
            cfg_parseable = False
    row("config.toml", "[OK]" if (cfg_exists and cfg_parseable) else ("[--]" if not cfg_exists else "[NG]"),
        str(cfg_file))

    # memories_dir
    try:
        settings = get_settings()
        memories_dir = settings.memories_dir
    except Exception:
        settings = None
        memories_dir = home / "memories"

    memories_accessible = memories_dir.is_dir()
    md_count = 0
    if memories_accessible:
        trash_dir = memories_dir / "_trash"
        for f in memories_dir.rglob("*.md"):
            try:
                f.relative_to(trash_dir)
            except ValueError:
                md_count += 1
    row("memories_dir access", "[OK]" if memories_accessible else "[--]",
        f"{memories_dir}  ({md_count} files)" if memories_accessible else str(memories_dir))

    # index.db
    db_path = home / "index.db"
    db_exists = db_path.is_file()
    db_size = db_path.stat().st_size if db_exists else 0
    row("index.db", "[OK]" if db_exists else "[--]",
        f"{db_size:,} bytes" if db_exists else "not created")

    print()

    # Embedding model cache
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        hf_cache = Path(hf_home) / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    embed_cache = False
    if hf_cache.is_dir():
        # Match the cache dir of the *configured* model (default: all-MiniLM-L6-v2)
        embed_model = (
            settings.embed_model if settings
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        model_prefix = "models--" + embed_model.replace("/", "--")
        embed_cache = any(
            d.name.startswith(model_prefix)
            for d in hf_cache.iterdir()
            if d.is_dir()
        )
    row("Embedding model cache", "[OK]" if embed_cache else "[--]",
        "present" if embed_cache else f"not downloaded ({hf_cache})")

    # Embedding runtime backend (ONNX is default; falls back to torch, which is heavy, if not generated)
    try:
        _settings_for_backend = get_settings()
        backend_status, backend_detail = check_embed_backend(_settings_for_backend)
    except Exception as e:
        backend_status, backend_detail = "[NG]", f"failed to load settings: {e}"
    row("Embedding runtime backend", backend_status, backend_detail)

    print()

    # engram-mcp
    mcp_path = get_engram_mcp_path()
    row("engram-mcp", "[OK]" if mcp_path else "[NG]",
        str(mcp_path) if mcp_path else "not found")

    print()

    # Claude Code
    claude_installed = shutil.which("claude") is not None
    if not claude_installed:
        row("Claude Code CLI", "[--]", "not installed")
        row("Claude Code MCP registration", "[--]", "Claude Code not installed")
    else:
        row("Claude Code CLI", "[OK]", shutil.which("claude") or "")
        registered = is_claude_mcp_registered()
        row("Claude Code MCP registration", "[OK]" if registered else "[NG]",
            "registered" if registered else "not registered (please run `engram setup`)")

    # CLAUDE.md
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.is_file():
        has_engram = "engram" in claude_md.read_text(encoding="utf-8", errors="replace")
        row("CLAUDE.md engram integration", "[OK]" if has_engram else "[NG]",
            str(claude_md) if has_engram else f"not appended: {claude_md}")
    else:
        row("CLAUDE.md engram integration", "[--]", f"file not found: {claude_md}")

    # Hook registration (auto-encoding / spontaneous recall)
    claude_settings = Path.home() / ".claude" / "settings.json"
    if claude_settings.is_file():
        try:
            sdata = json.loads(
                claude_settings.read_text(encoding="utf-8", errors="replace")
            )
            hk = sdata.get("hooks", {})

            def _has_engram_hook(event: str) -> bool:
                for entry in hk.get(event, []):
                    for h in entry.get("hooks", []):
                        if _is_engram_hook_cmd(h.get("command", "")):
                            return True
                return False

            se = _has_engram_hook("SessionEnd")
            up = _has_engram_hook("UserPromptSubmit")
            row("Auto-encoding hook (SessionEnd)", "[OK]" if se else "[NG]",
                "registered" if se else "not registered (please run `engram setup`)")
            row("Spontaneous recall hook (UserPromptSubmit)", "[OK]" if up else "[NG]",
                "registered" if up else "not registered (please run `engram setup`)")
        except Exception:
            row("Hook registration", "[NG]", "settings.json read error")
    else:
        row("Hook registration", "[--]", f"file not found: {claude_settings}")

    # Spontaneous recall mode and log
    try:
        _settings = get_settings()
        mode = _settings.surface_mode
        surface_log = _settings.data_dir / "surface" / "surface_log.jsonl"
        if surface_log.is_file():
            with surface_log.open("r", encoding="utf-8", errors="replace") as f:
                n_log = sum(1 for _ in f)
            detail = f"{n_log} log entries"
        else:
            detail = "no log"
        row("surface_mode", "[OK]", f"{mode} ({detail})")
    except Exception:
        row("surface_mode", "[NG]", "settings read error")

    # perf log summary (recent recall p50 / preload). Diagnoses perceived tool-call speed
    try:
        _settings_perf = get_settings()
        perf_log_path = _settings_perf.data_dir / "perf" / "perf_log.jsonl"
        if not _settings_perf.perf_log:
            row("perf log", "[--]", "disabled (settings.perf_log=false)")
        else:
            perf_status, perf_detail = summarize_perf(perf_log_path)
            row("perf log", perf_status, perf_detail)
    except Exception:
        row("perf log", "[NG]", "settings read error")

    print()

    # Codex
    codex_cfg = Path.home() / ".codex" / "config.toml"
    if codex_cfg.is_file():
        try:
            text = codex_cfg.read_text(encoding="utf-8")
            registered = "[mcp_servers.engram]" in text
            row("Codex MCP registration", "[OK]" if registered else "[NG]",
                "registered" if registered else "not registered")
        except Exception:
            row("Codex MCP registration", "[NG]", "read error")
    else:
        row("Codex MCP registration", "[--]", "Codex not installed")

    # AGENTS.md
    agents_md = Path.home() / ".codex" / "AGENTS.md"
    if agents_md.is_file():
        has_engram = "engram" in agents_md.read_text(encoding="utf-8", errors="replace")
        row("AGENTS.md engram integration", "[OK]" if has_engram else "[NG]",
            "integrated" if has_engram else "not appended")
    else:
        row("AGENTS.md engram integration", "[--]", "file not found")

    print()

    # Antigravity / Gemini
    gemini_cfg = Path.home() / ".gemini" / "config" / "mcp_config.json"
    if gemini_cfg.is_file():
        try:
            data = json.loads(gemini_cfg.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            registered = "engram" in servers
            row("Antigravity MCP registration", "[OK]" if registered else "[NG]",
                "registered" if registered else "not registered")
        except Exception:
            row("Antigravity MCP registration", "[NG]", "JSON read error")
    else:
        row("Antigravity MCP registration", "[--]", "Antigravity not installed")

    # GEMINI.md
    gemini_md = Path.home() / ".gemini" / "GEMINI.md"
    if gemini_md.is_file():
        has_engram = "engram" in gemini_md.read_text(encoding="utf-8", errors="replace")
        row("GEMINI.md engram integration", "[OK]" if has_engram else "[NG]",
            "integrated" if has_engram else "not appended")
    else:
        row("GEMINI.md engram integration", "[--]", "file not found")

    print()
