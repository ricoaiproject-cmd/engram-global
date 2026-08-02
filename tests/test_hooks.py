"""Tests for the hooks (auto-encoding and spontaneous recall) and hook registration."""

from __future__ import annotations

import json

import pytest

from engram.config import get_settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.hooks import (
    _consolidation_nudge,
    _read_consolidation_state,
    _skill_nudge,
    _write_consolidation_state,
    run_session_end,
    run_user_prompt,
)
from engram.setup import (
    merge_config_toml,
    read_config_toml,
    register_claude_hooks,
    write_config_toml,
)
from engram.store import MarkdownStore

DAY = 86400.0
NOW = 1_750_000_000.0


@pytest.fixture
def engram_home(tmp_path, monkeypatch):
    """Point ENGRAM_HOME at a temp directory (isolating settings, DB, and memory)."""
    home = tmp_path / "engram_home"
    home.mkdir()
    monkeypatch.setenv("ENGRAM_HOME", str(home))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    monkeypatch.delenv("ENGRAM_DATA_DIR", raising=False)
    return home


def _fake_build_engine(settings=None, *, embedder=None):
    settings = settings or get_settings()
    embedder = embedder or FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    return MemoryEngine(settings=settings, store=store, db=db,
                        embedder=embedder)


def _write_transcript(path):
    objs = [
        {"type": "summary", "summary": "計画書の作成"},
        {"type": "user", "cwd": "C:/proj/demo",
         "message": {"role": "user",
                     "content": "来年度の委員会計画書のたたき台を作ってほしい"}},
        {"type": "assistant",
         "message": {"role": "assistant",
                     "content": [{"type": "text",
                                  "text": "完成させました。"}]}},
    ]
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# session-end (auto-encoding)
# ---------------------------------------------------------------------------

def test_session_end_creates_episode(engram_home, tmp_path, monkeypatch):
    import engram.engine
    monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript)
    stdin = json.dumps({
        "session_id": "sess-1",
        "transcript_path": str(transcript),
        "cwd": "C:/proj/demo",
    })

    assert run_session_end(stdin) == 0

    settings = get_settings()
    db = IndexDB(settings.db_path, 64)
    episodes = db.all_memories(types=["episode"])
    assert len(episodes) == 1
    db.close()

    # Recorded as already encoded
    marks = (engram_home / "encoded_sessions.txt").read_text(encoding="utf-8")
    assert "sess-1" in marks

    # Don't encode the same session twice
    assert run_session_end(stdin) == 0
    db = IndexDB(settings.db_path, 64)
    assert len(db.all_memories(types=["episode"])) == 1
    db.close()


def test_session_end_skips_trivial_transcript(engram_home, tmp_path,
                                              monkeypatch):
    import engram.engine
    monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user",
                    "message": {"role": "user", "content": "ok"}},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stdin = json.dumps({"session_id": "sess-2",
                        "transcript_path": str(transcript), "cwd": ""})
    assert run_session_end(stdin) == 0
    settings = get_settings()
    assert not settings.db_path.is_file() or not IndexDB(
        settings.db_path, 64
    ).all_memories(types=["episode"])


def test_session_end_never_raises_on_garbage(engram_home):
    assert run_session_end("not json at all") == 0
    assert run_session_end("{}") == 0


# ---------------------------------------------------------------------------
# user-prompt (spontaneous recall)
# ---------------------------------------------------------------------------

def test_user_prompt_shadow_logs(engram_home, monkeypatch):
    # Set up one memory
    engine = _fake_build_engine()
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    stdin = json.dumps({
        "session_id": "sess-3",
        "prompt": "予算要求の書式ってどうだったっけ",
        "cwd": "C:/anywhere",
    })
    assert run_user_prompt(stdin) == 0

    log = engram_home / "surface" / "surface_log.jsonl"
    assert log.is_file()
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["mode"] == "shadow"


def test_user_prompt_active_outputs_context(engram_home, monkeypatch, capsys):
    (engram_home / "config.toml").write_text(
        "surface_mode = 'active'\nsurface_threshold = 0.3\n",
        encoding="utf-8",
    )
    engine = _fake_build_engine()
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    stdin = json.dumps({
        "session_id": "sess-4",
        "prompt": "予算要求の書式ってどうだったっけ",
        "cwd": "C:/anywhere",
    })
    assert run_user_prompt(stdin) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert "様式7" in ctx
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_user_prompt_skips_short_and_slash(engram_home):
    assert run_user_prompt(json.dumps({
        "session_id": "s", "prompt": "短い", "cwd": ""})) == 0
    assert run_user_prompt(json.dumps({
        "session_id": "s", "prompt": "/clear して再開しよう", "cwd": ""})) == 0
    assert not (engram_home / "surface" / "surface_log.jsonl").exists()


def test_user_prompt_skips_system_text(engram_home):
    # Does not react to non-human text injected by an agent/harness
    for noise in (
        "<task-notification>\n<task-id>abc</task-id>\n</task-notification>",
        "<!-- attach -->\n> 過去の引用文がここに入る",
        "<command-name>/model</command-name>",
        "Caveat: The messages below were generated by the user",
        "[Request interrupted by user]",
    ):
        assert run_user_prompt(json.dumps({
            "session_id": "s", "prompt": noise, "cwd": ""})) == 0
    assert not (engram_home / "surface" / "surface_log.jsonl").exists()


# ---------------------------------------------------------------------------
# consolidation nudge (automatic consolidation prompt)
# ---------------------------------------------------------------------------

class TestConsolidationNudge:
    """Unit tests for _consolidation_nudge (state file read/write only, no engine)."""

    def test_nudge_fires_when_clusters_at_threshold(self, engram_home):
        """A nudge message is returned when the cluster count is at/above the threshold and the last nudge is old enough."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 3,  # same as the default value of consolidate_nudge_min_clusters
            "last_nudged_at": NOW - 30 * DAY,  # old enough
        })

        msg = _consolidation_nudge(settings, now=NOW)
        assert msg is not None
        assert "3" in msg
        assert "consolidation_candidates" in msg
        assert "mark_consolidated" in msg

    def test_nudge_stamps_last_nudged_at(self, engram_home):
        """last_nudged_at is updated when a nudge message is returned."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 5, "last_nudged_at": 0.0,
        })

        msg = _consolidation_nudge(settings, now=NOW)
        assert msg is not None

        state = _read_consolidation_state(settings)
        assert state["last_nudged_at"] == NOW

    def test_nudge_throttled_on_second_immediate_call(self, engram_home):
        """A call right after a nudge returns None (throttled by the minimum interval)."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 4, "last_nudged_at": NOW - 30 * DAY,
        })

        first = _consolidation_nudge(settings, now=NOW)
        assert first is not None

        # Second call right after (last_nudged_at was just stamped)
        second = _consolidation_nudge(settings, now=NOW + 1.0)
        assert second is None

    def test_no_nudge_below_cluster_threshold(self, engram_home):
        """Does not nudge when the cluster count is below the threshold."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 2,  # below min_clusters (default 3)
            "last_nudged_at": NOW - 30 * DAY,
        })

        assert _consolidation_nudge(settings, now=NOW) is None

    def test_no_nudge_when_interval_not_elapsed(self, engram_home):
        """Does not nudge when the minimum interval (default 7 days) has not elapsed."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 5,
            "last_nudged_at": NOW - 1 * DAY,  # less than 7 days have passed
        })

        assert _consolidation_nudge(settings, now=NOW) is None

    def test_no_nudge_when_setting_disabled(self, engram_home):
        """Does not nudge when consolidate_nudge=False, even if the threshold and interval are met."""
        (engram_home / "config.toml").write_text(
            "consolidate_nudge = false\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 10, "last_nudged_at": NOW - 30 * DAY,
        })

        assert _consolidation_nudge(settings, now=NOW) is None

    def test_nudge_works_even_when_surface_mode_off(self, engram_home):
        """The nudge itself still fires independently even when surface_mode='off'."""
        (engram_home / "config.toml").write_text(
            "surface_mode = 'off'\n", encoding="utf-8",
        )
        settings = get_settings()
        assert settings.surface_mode == "off"
        _write_consolidation_state(settings, {
            "clusters": 3, "last_nudged_at": NOW - 30 * DAY,
        })

        assert _consolidation_nudge(settings, now=NOW) is not None


class TestConsolidationNudgeViaUserPrompt:
    """Wiring test for the nudge via run_user_prompt (reflected in additionalContext)."""

    def test_user_prompt_emits_nudge_when_surface_mode_off(self, engram_home,
                                                            capsys):
        """The nudge produces additionalContext on its own even with
        surface_mode=off and no surface-derived context."""
        (engram_home / "config.toml").write_text(
            "surface_mode = 'off'\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 3, "last_nudged_at": 0.0,
        })

        stdin = json.dumps({
            "session_id": "sess-nudge-1",
            "prompt": "次の作業を始めましょうか",
            "cwd": "C:/anywhere",
        })
        assert run_user_prompt(stdin) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "consolidation_candidates" in ctx
        assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"

    def test_user_prompt_no_nudge_below_threshold(self, engram_home, capsys):
        """No additionalContext is produced when the cluster count is below
        the threshold (surface is also irrelevant here, so nothing is output)."""
        (engram_home / "config.toml").write_text(
            "surface_mode = 'off'\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 1, "last_nudged_at": 0.0,
        })

        stdin = json.dumps({
            "session_id": "sess-nudge-2",
            "prompt": "次の作業を始めましょうか",
            "cwd": "C:/anywhere",
        })
        assert run_user_prompt(stdin) == 0
        out = capsys.readouterr().out
        assert out.strip() == ""


class TestSessionEndWritesConsolidationState:
    def test_run_session_end_writes_state_file(self, engram_home, tmp_path,
                                                monkeypatch):
        """run_session_end writes consolidation_state.json after auto-encoding,
        and it has a clusters key (when consolidate_nudge defaults to True)."""
        import engram.engine
        monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        stdin = json.dumps({
            "session_id": "sess-nudge-state",
            "transcript_path": str(transcript),
            "cwd": "C:/proj/demo",
        })

        assert run_session_end(stdin) == 0

        settings = get_settings()
        state_path = settings.data_dir / "consolidation_state.json"
        assert state_path.is_file()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "clusters" in state
        assert "checked_at" in state


# ---------------------------------------------------------------------------
# skill nudge (automatic prompt for skill-extraction candidates)
# ---------------------------------------------------------------------------

class TestSkillNudge:
    """Unit tests for _skill_nudge (state file read/write only, no engine)."""

    def test_nudge_fires_when_clusters_at_threshold(self, engram_home):
        """A nudge message is returned when the cluster count is at/above the threshold and the last nudge is old enough."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "skill_clusters": 1,  # same as the default value of skill_nudge_min_clusters
            "last_skill_nudged_at": NOW - 30 * DAY,
        })

        msg = _skill_nudge(settings, now=NOW)
        assert msg is not None
        assert "1" in msg
        assert "skill_candidates" in msg
        assert "mark_consolidated" in msg

    def test_no_nudge_when_setting_disabled(self, engram_home):
        """Does not nudge when skill_nudge=False, even if the threshold and interval are met."""
        (engram_home / "config.toml").write_text(
            "skill_nudge = false\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "skill_clusters": 10, "last_skill_nudged_at": NOW - 30 * DAY,
        })

        assert _skill_nudge(settings, now=NOW) is None

    def test_no_nudge_below_cluster_threshold(self, engram_home):
        """Does not nudge when the cluster count is below the threshold."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "skill_clusters": 0,  # below min_clusters (default 1)
            "last_skill_nudged_at": NOW - 30 * DAY,
        })

        assert _skill_nudge(settings, now=NOW) is None

    def test_no_nudge_when_interval_not_elapsed(self, engram_home):
        """Does not nudge when the minimum interval (default 7 days) has not elapsed (suppresses re-nudging)."""
        settings = get_settings()
        _write_consolidation_state(settings, {
            "skill_clusters": 2,
            "last_skill_nudged_at": NOW - 1 * DAY,  # less than 7 days have passed
        })

        assert _skill_nudge(settings, now=NOW) is None


class TestSessionEndWritesSkillClusters:
    def test_run_session_end_writes_skill_clusters(self, engram_home, tmp_path,
                                                     monkeypatch):
        """run_session_end writes skill_clusters to the state file after
        auto-encoding (when skill_nudge defaults to True)."""
        import engram.engine
        monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        stdin = json.dumps({
            "session_id": "sess-skill-state",
            "transcript_path": str(transcript),
            "cwd": "C:/proj/demo",
        })

        assert run_session_end(stdin) == 0

        settings = get_settings()
        state_path = settings.data_dir / "consolidation_state.json"
        assert state_path.is_file()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "skill_clusters" in state


class TestSkillNudgeViaUserPrompt:
    """Wiring test for the skill-extraction-candidate nudge via run_user_prompt."""

    def test_user_prompt_emits_skill_nudge(self, engram_home, capsys):
        (engram_home / "config.toml").write_text(
            "surface_mode = 'off'\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "skill_clusters": 2, "last_skill_nudged_at": 0.0,
        })

        stdin = json.dumps({
            "session_id": "sess-skill-nudge-1",
            "prompt": "次の作業を始めましょうか",
            "cwd": "C:/anywhere",
        })
        assert run_user_prompt(stdin) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "skill_candidates" in ctx

    def test_consolidation_and_skill_nudge_both_present(self, engram_home, capsys):
        """When both the consolidation and skill-extraction nudge conditions
        are met, both appear in additionalContext."""
        (engram_home / "config.toml").write_text(
            "surface_mode = 'off'\n", encoding="utf-8",
        )
        settings = get_settings()
        _write_consolidation_state(settings, {
            "clusters": 3, "last_nudged_at": 0.0,
            "skill_clusters": 2, "last_skill_nudged_at": 0.0,
        })

        stdin = json.dumps({
            "session_id": "sess-both-nudge",
            "prompt": "次の作業を始めましょうか",
            "cwd": "C:/anywhere",
        })
        assert run_user_prompt(stdin) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "consolidation_candidates" in ctx
        assert "skill_candidates" in ctx


# ---------------------------------------------------------------------------
# Hook registration (setup)
# ---------------------------------------------------------------------------

def test_register_claude_hooks_creates_file(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    ok, msg = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    events = data["hooks"]
    assert "SessionEnd" in events
    assert "UserPromptSubmit" in events
    cmd = events["SessionEnd"][0]["hooks"][0]["command"]
    assert "engram" in cmd and "hook session-end" in cmd


def test_register_claude_hooks_idempotent(tmp_path):
    settings_path = tmp_path / "settings.json"
    exe = tmp_path / "engram.exe"
    register_claude_hooks(settings_path, exe)
    before = settings_path.read_text(encoding="utf-8")
    ok, msg = register_claude_hooks(settings_path, exe)
    assert ok
    assert "skipped" in msg
    assert settings_path.read_text(encoding="utf-8") == before


def test_register_claude_hooks_updates_stale_path(tmp_path):
    settings_path = tmp_path / "settings.json"
    old_exe = tmp_path / "old" / "engram.exe"
    new_exe = tmp_path / "new" / "engram.exe"
    register_claude_hooks(settings_path, old_exe)
    ok, msg = register_claude_hooks(settings_path, new_exe)
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    cmd = data["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    # Compare full paths (a substring check for "old" was observed to
    # false-positive on macOS temp paths like /var/folders/...)
    assert str(new_exe) in cmd and str(old_exe) not in cmd
    # No duplicate entries were created
    assert len(data["hooks"]["SessionEnd"]) == 1


def test_register_claude_hooks_preserves_other_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {
            "SessionEnd": [
                {"hooks": [{"type": "command", "command": "other-tool run"}]}
            ]
        },
    }), encoding="utf-8")
    ok, _ = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    cmds = [h["command"] for e in data["hooks"]["SessionEnd"]
            for h in e["hooks"]]
    assert "other-tool run" in cmds
    assert any("engram" in c for c in cmds)
    assert data["permissions"]["allow"] == ["Bash(ls:*)"]


def test_register_claude_hooks_refuses_broken_json(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{broken json", encoding="utf-8")
    ok, msg = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert not ok
    assert settings_path.read_text(encoding="utf-8") == "{broken json"


# ---------------------------------------------------------------------------
# config.toml round-trip of a dict (section)
# ---------------------------------------------------------------------------

def test_config_toml_roundtrip_with_room_paths(tmp_path):
    cfg = tmp_path / "config.toml"
    write_config_toml(cfg, {
        "memories_dir": "C:/mem",
        "room_paths": {"C:/Users/me/work": "work",
                       "H:/マイドライブ/個人": "personal"},
    })
    data = read_config_toml(cfg)
    assert data["memories_dir"] == "C:/mem"
    assert data["room_paths"]["C:/Users/me/work"] == "work"
    assert data["room_paths"]["H:/マイドライブ/個人"] == "personal"

    # The section is preserved even after merging
    merge_config_toml(cfg, {"surface_mode": "active"})
    data = read_config_toml(cfg)
    assert data["surface_mode"] == "active"
    assert data["room_paths"]["C:/Users/me/work"] == "work"


# ---------------------------------------------------------------------------
# The mark_consolidated tool immediately updates the nudge state (via server)
# ---------------------------------------------------------------------------


class TestMarkConsolidatedRefreshesState:
    def test_state_refreshed_after_mark_consolidated(self, engram_home, monkeypatch):
        import engram.server as server
        from engram.hooks import (
            _read_consolidation_state,
            _write_consolidation_state,
        )

        settings = get_settings()

        class _FakeEngine:
            def __init__(self):
                self.settings = settings

            def mark_consolidated(self, episode_ids, new_memory_id):
                return {
                    "consolidated": episode_ids,
                    "new_memory_id": new_memory_id,
                    "status": "ok",
                }

            def consolidation_candidates(self):
                # Assumes only 1 cluster remains after consolidation
                return {"clusters": [{"ids": ["a", "b"], "contents": ["x", "y"]}]}

            def skill_candidates(self):
                # Assumes only 1 cluster remains for skill candidates too
                return {"clusters": [{"ids": ["c", "d"], "contents": ["y", "z"]}]}

        monkeypatch.setattr(server, "_engine", _FakeEngine())
        # Prior state: stale cluster count from before consolidation
        _write_consolidation_state(
            settings, {"clusters": 5, "skill_clusters": 9, "checked_at": 0.0}
        )

        result = server.mark_consolidated(["e1", "e2"], "new1")

        assert result["status"] == "ok"
        state = _read_consolidation_state(settings)
        assert state["clusters"] == 1  # not left stuck at 5
        assert state["skill_clusters"] == 1  # not left stuck at 9 (skill candidates side is also updated)
