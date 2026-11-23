"""The activity ledger is built from runtime events only: what it summarises is what ran."""

from __future__ import annotations

import pytest

from hermes_easy import activity
from hermes_easy.activity import Ledger, category_of, summarize


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(activity, "CURRENT", None)


def run(led, name, args=None, *, failed=False, result='{"output": "ok", "exit_code": 0}', duration=0.5, neutral=False):
    led.tool_started(name, args or {}, label=name)
    return led.tool_completed(name, result, failed=failed, neutral=neutral, duration=duration)


class TestCategories:
    @pytest.mark.parametrize("command,expected", [
        ("pytest -q tests/unit", "tests"), ("npm run test", "tests"), ("vitest run", "tests"),
        ("scripts/run_tests.sh tests/a.py", "tests"), ("cargo test --release", "tests"),
        ("ls -la", "shell"), ("git status", "shell"), ("make build", "shell"),
    ])
    def test_a_terminal_command_is_shell_or_tests(self, command, expected):
        assert category_of("terminal", {"command": command}) == expected

    def test_other_tools(self):
        assert category_of("read_file") == "read"
        assert category_of("patch") == category_of("write_file") == "edit"
        assert category_of("search_files") == "search"
        assert category_of("web_search") == "web"
        assert category_of("browser_click") == "browser"
        assert category_of("delegate_task") == "agent"
        assert category_of("something_new") == "other"


class TestSummary:
    def test_the_phrases_the_user_reads(self):
        led = Ledger()
        for i in range(4):
            run(led, "terminal", {"command": f"echo {i}"})
        for i in range(6):
            run(led, "read_file", {"path": f"/p/f{i}.py"})
        for name in ("a.py", "b.py", "c.py"):
            run(led, "patch", {"path": f"/p/{name}"})
        group = led.close_group()
        assert summarize(group) == "Ran 4 shell commands · Read 6 files · Edited 3 files"

    def test_files_are_counted_once_however_often_they_are_touched(self):
        led = Ledger()
        for _ in range(3):
            run(led, "read_file", {"path": "/p/same.py"})
        assert summarize(led.close_group()) == "Read 1 file"

    def test_singular_tests_search_browser_and_present_tense(self):
        led = Ledger()
        run(led, "terminal", {"command": "pytest"})
        run(led, "search_files", {"pattern": "x"})
        run(led, "browser_navigate", {"url": "https://a.b"})
        group = led.close_group()
        assert summarize(group) == "Executed tests · Searched the repository · Used browser"
        assert summarize(group, active=True) == "Executing tests · Searching the repository · Using browser"

    def test_order_follows_first_use(self):
        led = Ledger()
        run(led, "read_file", {"path": "a"})
        run(led, "terminal", {"command": "ls"})
        run(led, "read_file", {"path": "b"})
        assert summarize(led.close_group()).startswith("Read 2 files · Ran 1 shell command")


class TestLifecycle:
    def test_a_started_call_is_running_until_the_runtime_completes_it(self):
        led = Ledger()
        call = led.tool_started("terminal", {"command": "sleep 5", "workdir": "/w"}, label="$ sleep 5")
        assert led.running_calls() == [call] and led.state == "tools"
        assert call.cwd == "/w" and call.machine
        led.tool_completed("terminal", '{"output":"", "exit_code": 0}', failed=False, duration=5.0)
        assert led.running_calls() == [] and led.state == "waiting"
        assert call.elapsed() == pytest.approx(5.0)

    def test_parallel_calls_close_in_order_and_the_first_is_found_by_name(self):
        led = Ledger()
        a = led.tool_started("terminal", {"command": "a"})
        b = led.tool_started("terminal", {"command": "b"})
        led.tool_completed("terminal", "ok", failed=False, duration=1)
        assert not a.running and b.running
        assert led.state == "tools"

    def test_a_completion_with_no_matching_start_is_ignored(self):
        assert Ledger().tool_completed("terminal", "ok", failed=False) is None

    def test_failures_are_counted_but_a_neutral_failure_is_an_answer(self):
        led = Ledger()
        run(led, "terminal", {"command": "false"}, failed=True,
            result='{"output": "boom", "exit_code": 1, "error": "bad"}')
        run(led, "search_files", {"pattern": "zzz"}, failed=True, neutral=True)
        assert led.failures_this_turn == 1
        group = led.close_group()
        assert [c.exit_code for c in group.failed] == [1]
        assert group.failed[0].error == "bad" and group.failed[0].output == "boom"

    def test_close_group_keeps_history_bounded_and_returns_none_when_empty(self):
        led = Ledger()
        assert led.close_group() is None
        for _ in range(Ledger.HISTORY + 5):
            run(led, "terminal", {"command": "x"})
            led.close_group()
        assert len(led.groups) == Ledger.HISTORY

    def test_a_call_the_runtime_never_completed_is_reported_as_interrupted(self):
        led = Ledger()
        led.tool_started("terminal", {"command": "hang"})
        group = led.close_group()
        call = group.calls[0]
        assert call.failed and call.error == "Interrupted before it finished" and not call.running
        assert group.failed == [call]

    def test_claim_completed_hands_each_call_out_exactly_once(self):
        led = Ledger()
        run(led, "terminal", {"command": "a"})
        assert led.claim_completed("terminal") is not None
        assert led.claim_completed("terminal") is None

    def test_reasoning_is_active_only_while_deltas_keep_arriving(self):
        led = Ledger()
        led.note_reasoning(now=100.0)
        assert led.reasoning_active(now=101.0) and not led.reasoning_active(now=110.0)

    def test_a_new_turn_forgets_the_old_one(self):
        led = Ledger()
        run(led, "terminal", {"command": "x"})
        led.start_turn(now=50.0)
        assert led.calls_this_turn == 0 and led.current is None and led.state == "thinking"


class TestPersistence:
    def test_history_survives_a_reload(self, tmp_path):
        led = Ledger()
        run(led, "terminal", {"command": "pytest"}, failed=True, result='{"output":"x","exit_code":2}')
        led.close_group()
        activity.save(led, "sess-1")
        again = Ledger()
        activity.load(again, "sess-1")
        assert summarize(again.groups[0]) == "Executed tests"
        assert again.groups[0].calls[0].exit_code == 2
        assert again.next_id > 1

    def test_a_missing_or_corrupt_file_is_just_empty(self, tmp_path):
        led = Ledger()
        activity.load(led, "nope")
        path = activity._store_path("bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        activity.load(led, "bad")
        assert led.groups == []


class TestMachines:
    def test_local_names_the_os_and_remote_names_the_host(self, monkeypatch):
        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "ssh", "ssh_host": "build-box"})
        assert activity.machine_label() == "remote server build-box"
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local"})
        monkeypatch.setattr(activity.platform, "system", lambda: "Darwin")
        assert activity.machine_label() == "macOS"
        monkeypatch.setattr(activity.platform, "system", lambda: "Windows")
        assert activity.machine_label() == "Windows"
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "docker"})
        assert activity.machine_label() == "Docker container on Windows"

    def test_a_task_override_wins_over_the_global_backend(self, monkeypatch):
        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local"})
        monkeypatch.setattr(tt, "_qualify_task_key", lambda t: t)
        monkeypatch.setattr(tt, "_task_env_overrides", {"child-1": {"env_type": "ssh", "ssh_host": "win-runner"}})
        assert activity.machine_label("child-1") == "remote server win-runner"
        assert activity.machine_label("child-2") != "remote server win-runner"

    def test_two_agents_on_different_machines_keep_separate_labels(self, monkeypatch):
        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local"})
        monkeypatch.setattr(tt, "_qualify_task_key", lambda t: t)
        monkeypatch.setattr(tt, "_task_env_overrides", {"a": {"env_type": "ssh", "ssh_host": "win"}})
        led = Ledger()
        one = led.tool_started("terminal", {"command": "dir"}, task_id="a")
        two = led.tool_started("terminal", {"command": "ls"}, task_id="b")
        assert one.machine == "remote server win" and two.machine != one.machine


class TestTruthfulSummaries:
    def test_a_refused_write_is_not_reported_as_an_edit(self):
        led = Ledger()
        run(led, "patch", {"path": "/p/a.py"})
        run(led, "write_file", {"path": "/p/b.py"}, failed=True, result='{"error": "Refusing to overwrite"}')
        assert summarize(led.close_group()) == "Edited 1 file"

    def test_when_every_edit_failed_the_summary_says_so(self):
        led = Ledger()
        run(led, "write_file", {"path": "/p/b.py"}, failed=True, result='{"error": "no"}')
        run(led, "read_file", {"path": "/p/missing.py"}, failed=True, result='{"error": "not found"}')
        assert summarize(led.close_group()) == "Failed to edit 1 file · Failed to read 1 file"

    def test_a_command_that_ran_and_failed_is_still_a_command_that_ran(self):
        led = Ledger()
        run(led, "terminal", {"command": "false"}, failed=True, result='{"exit_code": 1}')
        assert summarize(led.close_group()) == "Ran 1 shell command"
