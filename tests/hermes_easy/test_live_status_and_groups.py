"""The live status row and the grouped tool transcript: real values only, readable when narrow."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

import hermes_easy
from hermes_easy import activity, layout, status
from hermes_easy import settings as ui
from hermes_easy import theme


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv(hermes_easy.ENV_FLAG, "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ui, "_current", None)
    monkeypatch.setattr(ui, "_problems", [])
    monkeypatch.setattr(theme, "install", lambda settings=None: None)
    monkeypatch.setattr(activity, "CURRENT", None)
    monkeypatch.setattr(layout, "_last_block", {"kind": None})
    monkeypatch.setattr(layout, "_hint_shown", {"done": False})


def strip(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def make_cli(width=100, running=True, **extra):
    agent = SimpleNamespace(session_input_tokens=0, session_output_tokens=0,
                            context_compressor=SimpleNamespace(context_length=65536, last_prompt_tokens=0))
    cli = SimpleNamespace(model="qwen3-coder-next", agent=agent, _agent_running=running, _prompt_start_time=1000.0,
                          session_id="s1", _get_tui_terminal_width=lambda: width,
                          _trim_status_bar_text=lambda t, n: t if len(t) <= n else t[:n - 1] + "…",
                          _status_bar_display_width=len)
    cli.__dict__.update(extra)
    return cli


def done(cli, name, args, result='{"output": "ok", "exit_code": 0}', *, failed=False, duration=1.0):
    from agent.display import get_tool_row_parts
    led = layout.ledger_for(cli)
    parts = get_tool_row_parts(name, args)[0]
    led.tool_started(name, args, label=f"{parts[1]} {parts[2]}".strip())
    led.tool_completed(name, result, failed=failed, duration=duration)
    return layout.tool_row(cli, name, args, duration, result, is_error=failed)


class TestPlayfulWord:
    def test_it_rotates_and_differs_between_turns(self):
        words = {status.playful_word(1000, now=1000 + i * 2.7) for i in range(10)}
        assert len(words) >= 5
        assert status.playful_word(1000, now=1000.5) == status.playful_word(1000, now=1001.5)  # stable between ticks
        assert status.playful_word(1001, now=1000.5) != status.playful_word(1000, now=1000.5)

    def test_it_is_never_the_status(self):
        cli = make_cli()
        snap = status.snapshot(cli, now=1003.0)
        assert snap.word and snap.word not in {f[2] for f in status.fields(snap)}
        assert snap.status == "Thinking"


class TestFieldsAreReal:
    def test_a_fresh_turn_shows_status_time_and_model_only(self):
        cli = make_cli()
        snap = status.snapshot(cli, now=1004.0)
        keys = [k for k, _r, _t in status.fields(snap)]
        assert keys == ["status", "model"]
        assert status.fields(snap)[0][2] == "Thinking · 4s"

    def test_tokens_and_context_come_from_the_provider_usage(self):
        cli = make_cli()
        layout.ledger_for(cli)
        cli.agent.session_input_tokens, cli.agent.session_output_tokens = 12400, 1100
        cli.agent.context_compressor.last_prompt_tokens = 31457
        snap = status.snapshot(cli, now=1010.0)
        by_key = {k: t for k, _r, t in status.fields(snap)}
        assert by_key["tokens"] == "↑12.4k ↓1.1k" and by_key["context"] == "ctx 48%"

    def test_tokens_are_this_turn_not_the_whole_session(self):
        cli = make_cli()
        cli.agent.session_input_tokens = 50000
        layout.ledger_for(cli)  # the turn starts here: 50,000 were spent before it
        cli.agent.session_input_tokens = 52000
        assert status.snapshot(cli, now=1005.0).tokens_in == 2000

    @pytest.mark.parametrize("used,role", [(20000, "dim"), (50000, "warn"), (62000, "error")])
    def test_context_colour_follows_pressure(self, used, role):
        cli = make_cli()
        cli.agent.context_compressor.last_prompt_tokens = used
        fields = {k: r for k, r, _t in status.fields(status.snapshot(cli, now=1001.0))}
        assert fields["context"] == role

    def test_a_running_command_is_executing_with_its_machine(self):
        cli = make_cli()
        layout.ledger_for(cli).tool_started("terminal", {"command": "pytest"}, label="$ pytest")
        snap = status.snapshot(cli, now=1002.0)
        assert snap.status == "Executing terminal" and snap.tone == "tool" and snap.machine
        assert snap.tools == 1

    def test_failures_are_shown_in_the_tool_count(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "false"}, '{"output":"x","exit_code":1}', failed=True)
        text = {k: t for k, _r, t in status.fields(status.snapshot(cli, now=1005.0))}["tools"]
        assert text == "1 tool, 1 failed"

    def test_recovery_is_continuing_in_the_warning_colour(self):
        cli = make_cli()
        layout.ledger_for(cli).note_recovery("↻ Tool call cut off", now=1001.0)
        snap = status.snapshot(cli, now=1002.0)
        assert (snap.status, snap.tone) == ("Continuing", "warn")

    def test_waiting_for_an_approval_is_not_reported_as_working(self):
        cli = make_cli(_approval_state={"command": "rm"})
        assert status.snapshot(cli, now=1002.0).status == "Waiting for you"

    def test_reasoning_shows_only_while_the_provider_is_streaming_it(self):
        cli = make_cli()
        layout.ledger_for(cli).note_reasoning(now=1001.0)
        assert ("reasoning" in [k for k, _r, _t in status.fields(status.snapshot(cli, now=1002.0))])
        assert ("reasoning" not in [k for k, _r, _t in status.fields(status.snapshot(cli, now=1030.0))])

    def test_nothing_is_invented_without_an_agent(self):
        cli = make_cli()
        cli.agent = None
        snap = status.snapshot(cli, now=1001.0)
        assert snap.context_pct is None and snap.tokens_in == 0


class TestNarrow:
    def test_fields_drop_whole_and_the_status_survives(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "x"})
        cli.agent.session_input_tokens, cli.agent.session_output_tokens = 5000, 400
        cli.agent.context_compressor.last_prompt_tokens = 1000
        items = status.fields(status.snapshot(cli, now=1005.0))
        for width in (120, 60, 40, 24, 14):
            fit = status.fit(items, width, lead=2)
            assert fit[0][0] == "status"
            assert 2 + sum(len(t) for _k, _r, t in fit) + 2 * (len(fit) - 1) <= width

    def test_compact_numbers(self):
        assert [status.compact(n) for n in (0, 7, 999, 1000, 12400, 1_250_000)] == ["", "7", "999", "1k", "12.4k", "1.2M"]


class TestLiveRow:
    def text(self, cli):
        return "".join(t for _s, t in layout.live_row(cli))

    def test_the_row_leads_with_the_word_then_real_values(self):
        cli = make_cli()
        text = self.text(cli)
        assert re.search(r"[A-Z][a-z]+…  Thinking · \d+s", text) and "Qwen 3" not in text or "qwen" in text.lower()

    def test_the_word_can_be_switched_off_without_losing_the_status(self):
        s = ui.defaults()
        s["status"]["activity_word"] = False
        ui.replace(s)
        text = self.text(make_cli())
        assert "…" not in text and "Thinking" in text

    def test_running_commands_get_their_own_row_while_they_run(self):
        cli = make_cli()
        assert layout.live_tool_fragments(cli) == []
        layout.ledger_for(cli).tool_started("terminal", {"command": "pytest -q"}, label="$ pytest -q")
        row = "".join(t for _s, t in layout.live_tool_fragments(cli))
        assert "$ pytest -q" in row
        layout.ledger_for(cli).tool_completed("terminal", "ok", failed=False, duration=1)
        assert layout.live_tool_fragments(cli) == []

    def test_activity_fragments_hand_over_to_the_live_row_while_a_turn_runs(self):
        cli = make_cli(_easy_model_status=())
        assert "Thinking" in "".join(t for _s, t in layout.activity_fragments(cli))
        cli._agent_running = False
        assert layout.activity_fragments(cli) == []


class TestGroupedTranscript:
    def test_grouped_calls_print_nothing_until_the_group_closes(self):
        cli = make_cli()
        assert done(cli, "terminal", {"command": "ls"}) is None
        assert done(cli, "read_file", {"path": "/p/a.py"}) is None
        rows = layout.flush_group()
        text = strip("\n".join(rows))
        assert "Ran 1 shell command" in text and "Read 1 file" in text
        assert layout.flush_group() == []  # closed once

    def test_the_summary_is_green_when_everything_worked_and_red_when_not(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "ls"})
        ok = "\n".join(layout.flush_group())
        done(cli, "terminal", {"command": "false"}, '{"output":"boom","exit_code":1}', failed=True)
        bad = "\n".join(layout.flush_group())
        assert theme.hex_fg(theme.palette()["ok"]) in ok and "✗" not in strip(ok)
        assert "✗" in strip(bad) and theme.hex_fg(theme.palette()["error"], bold=True) in bad

    def test_a_failure_is_listed_under_the_summary_with_exit_status_and_error(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "pytest -q"}, '{"output": "E   assert 1 == 2\\nFAILED test_a.py", "exit_code": 1}', failed=True, duration=2.1)
        text = strip("\n".join(layout.flush_group()))
        assert "Executed tests" in text and "✗ $ pytest -q" in text and "exit 1 · 2.1s" in text
        assert "FAILED test_a.py" in text

    def test_an_answer_opening_flushes_the_group_first(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "ls"})
        opening = strip(layout.answer_open() or "")
        assert opening.lstrip().startswith("•") and "Ran 1 shell command" in opening

    def test_reasoning_and_the_turn_footer_flush_too(self):
        cli = make_cli()
        done(cli, "read_file", {"path": "/p/a.py"})
        assert "Read 1 file" in strip("\n".join(layout.thought_rows(1.0, "hm", False)))
        done(cli, "read_file", {"path": "/p/b.py"})
        assert "Read 1 file" in strip(layout.turn_footer("3s · read 1 file") or "")

    def test_rows_mode_still_prints_one_row_per_call(self):
        s = ui.defaults()
        s["layout"]["tool_groups"] = "rows"
        ui.replace(s)
        cli = make_cli()
        assert "ls" in strip(done(cli, "terminal", {"command": "ls"}) or "")
        assert layout.flush_group() == []

    def test_without_a_ledger_a_call_keeps_its_own_row(self):
        row = layout.tool_row(SimpleNamespace(), "terminal", {"command": "true"}, 0.1, '{"output": "", "exit_code": 0}')
        assert row and "true" in row

    def test_the_expand_hint_appears_once(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "a"})
        first = strip("\n".join(layout.flush_group()))
        done(cli, "terminal", {"command": "b"})
        second = strip("\n".join(layout.flush_group()))
        assert "Ctrl+O to expand" in first and "Ctrl+O to expand" not in second


class TestExpand:
    def test_ctrl_o_shows_every_call_with_command_directory_machine_time_and_exit(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "echo hi", "workdir": "/work/proj"}, '{"output": "hi", "exit_code": 0}', duration=0.4)
        done(cli, "terminal", {"command": "false"}, '{"output": "boom", "exit_code": 3}', failed=True, duration=1.2)
        layout.flush_group()
        text = strip("\n".join(layout.expand_last_tool_output(cli)))
        assert "✓ $ echo hi" in text and "✗ $ false" in text
        assert "/work/proj" in text and "0.4s" in text and "exit 3" in text and "boom" in text
        assert activity.machine_label().split()[0] in text

    def test_pressing_again_steps_back_one_group(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "first-cmd"})
        layout.flush_group()
        done(cli, "terminal", {"command": "second-cmd"})
        layout.flush_group()
        latest = strip("\n".join(layout.expand_last_tool_output(cli)))
        earlier = strip("\n".join(layout.expand_last_tool_output(cli)))
        assert "second-cmd" in latest and "first-cmd" in earlier

    def test_a_running_group_can_be_expanded_before_it_closes(self):
        cli = make_cli()
        done(cli, "terminal", {"command": "still-open"})
        assert "still-open" in strip("\n".join(layout.expand_last_tool_output(cli)))

    def test_failed_output_keeps_the_tail_where_the_error_is(self):
        cli = make_cli()
        out = "\\n".join(f"line {i}" for i in range(100))
        done(cli, "terminal", {"command": "build"}, f'{{"output": "{out}\\nFATAL: disk full", "exit_code": 2}}', failed=True)
        layout.flush_group()
        text = strip("\n".join(layout.expand_last_tool_output(cli)))
        assert "FATAL: disk full" in text and "line 0" not in text and "more lines" in text


class TestRecoveryNotice:
    def test_a_runtime_recovery_notice_reads_as_continuing_with_its_own_words(self):
        cli = make_cli()
        layout.ledger_for(cli).note_recovery("↻ tool call cut off, not run — retrying: asked for smaller pieces", now=1004.0)
        snap = status.snapshot(cli, now=1005.0)
        assert (snap.status, snap.tone) == ("Continuing", "warn")
        assert [t for k, _r, t in status.fields(snap) if k == "note"] == ["tool call cut off, not run — retrying: asked for smaller pieces"]

    def test_a_notice_expires_and_a_running_command_outranks_it(self):
        cli = make_cli()
        led = layout.ledger_for(cli)
        led.note_recovery("↻ cut off", now=1002.0)
        assert status.snapshot(cli, now=1030.0).status != "Continuing"
        led.note_recovery("↻ cut off", now=1029.0)
        led.tool_started("terminal", {"command": "ls"}, label="$ ls")
        assert status.snapshot(cli, now=1030.0).status == "Executing terminal"

    def test_the_cli_hooks_keep_the_notice_even_though_diagnostics_are_not_rendered(self, monkeypatch):
        from hermes_easy.cli_mixin import EasyUIMixin
        from gateway.warning_notifications import DiagnosticText
        cli = make_cli()
        cli._on_thinking = lambda text: None
        seen = []
        import hermes_easy.cli_mixin as mixin
        monkeypatch.setattr(mixin, "_upstream", lambda name: (lambda self, text: seen.append(text)))
        EasyUIMixin._on_thinking(cli, DiagnosticText("↻ reply hit the output limit — continuing (1/12)"))
        assert layout.ledger_for(cli).recovery.startswith("↻ reply hit the output limit") and seen


class TestEndOfTurn:
    def test_an_interrupted_turn_still_prints_its_open_group_with_the_interrupted_call(self, monkeypatch):
        import hermes_easy.cli_mixin as mixin
        printed = []
        import cli as climod
        monkeypatch.setattr(climod, "_cprint", lambda text="": printed.append(text))
        monkeypatch.setattr(mixin, "_upstream", lambda name: (lambda self: printed.append("<upstream footer>")))
        cli = make_cli()
        done(cli, "read_file", {"path": "/p/a.py"})
        layout.ledger_for(cli).tool_started("terminal", {"command": "sleep 100"}, label="$ sleep 100")
        mixin.EasyUIMixin._turn_summary_emit(cli)
        text = strip("\n".join(printed))
        assert "Read 1 file" in text and "Ran 1 shell command" in text
        assert "✗ $ sleep 100" in text and "Interrupted before it finished" in text
        assert printed[-1] == "<upstream footer>"


class TestParallelAgents:
    """Two subagents, two machines: each dock row names its own model and where its commands run."""

    def _monitor(self, monkeypatch):
        import tools.terminal_tool as tt
        import tools.delegate_tool_registry as reg
        from hermes_cli.cli_subagent_monitor import SubagentMonitor
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local"})
        monkeypatch.setattr(tt, "_qualify_task_key", lambda t: t)
        monkeypatch.setattr(tt, "_task_env_overrides", {
            "task-win": {"env_type": "ssh", "ssh_host": "win-runner"},
            "task-mac": {"env_type": "ssh", "ssh_host": "mac-mini"}})
        monkeypatch.setattr(reg, "_active_subagents", {
            "sa-1": {"agent": SimpleNamespace(task_id="task-win")},
            "sa-2": {"agent": SimpleNamespace(task_id="task-mac")},
            "sa-3": {"agent": SimpleNamespace(task_id="task-local")}})
        monitor = SubagentMonitor(SimpleNamespace())
        monitor.entries = [
            {"subagent_id": "sa-1", "goal": "build the Windows installer", "model": "claude-sonnet", "elapsed": 41, "status": "running", "last_tool": "terminal"},
            {"subagent_id": "sa-2", "goal": "run the macOS smoke test", "model": "qwen3-coder-next", "elapsed": 17, "status": "running", "last_tool": "read_file"},
            {"subagent_id": "sa-3", "goal": "lint", "model": "qwen3.8-27b", "elapsed": 3, "status": "running", "last_tool": ""},
        ]
        return monitor

    def test_each_row_carries_its_own_model_and_machine(self, monkeypatch):
        text = self._monitor(monkeypatch).dock_text(columns=200, rows=50)
        lines = {k: next(l for l in text.splitlines() if k in l) for k in ("Windows installer", "macOS smoke", "lint")}
        assert "claude-sonnet · remote server win-runner · 41s" in lines["Windows installer"]
        assert "qwen3-coder-next · remote server mac-mini · 17s" in lines["macOS smoke"]
        assert "remote server" not in lines["lint"] and "qwen3.8-27b" in lines["lint"]
        assert "terminal" in lines["Windows installer"] and "read_file" in lines["macOS smoke"]

    def test_stock_hermes_rows_are_unchanged(self, monkeypatch):
        monkeypatch.delenv(hermes_easy.ENV_FLAG)
        text = self._monitor(monkeypatch).dock_text(columns=200, rows=50)
        assert "remote server" not in text and "claude-sonnet" not in text
