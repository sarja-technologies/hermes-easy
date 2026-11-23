"""A tool call cut off mid-arguments is never run and never stored, and the turn recovers by sending a
*changed* request instead of the same one four times (the failure that ended long coding sessions with
"The model's reply was cut off ... Send `continue`").

Root cause these tests pin: a llama.cpp router reports the TRAINED window (262,144) while a slot holds
65,536, so the compressor never fired, the slot filled, and every retry truncated at the same place.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID

from tests.agent.test_run_agent import _mock_response, _mock_tool_call

NUDGE_MARK = "was NOT run"
FULL_SLOT = {"prompt_tokens": 63123, "completion_tokens": 2413, "total_tokens": 65536}
CAP_HIT = {"prompt_tokens": 2000, "completion_tokens": 4096, "total_tokens": 6096}


@pytest.fixture()
def agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a.valid_tool_names.add("write_file")
        return a


def _partial(call_id="c_bad", finish="length", usage=None, content=""):
    tc = _mock_tool_call(name="write_file", arguments='{"path":"big.py","content":"def f():\\n    return', call_id=call_id)
    return _mock_response(content=content, finish_reason=finish, tool_calls=[tc], usage=usage)


def _whole(call_id, path="big.py"):
    tc = _mock_tool_call(name="write_file", arguments=f'{{"path":"{path}","content":"x = 1"}}', call_id=call_id)
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])


def _run(agent, message="write big.py", executed=None):
    def fake(name, args, *a, **k):
        if executed is not None:
            executed.append((name, args))
        return '{"success": true}'

    with (
        patch("model_tools.handle_function_call", side_effect=fake),
        patch("agent.turn_tool_call_recovery._server_window", return_value=None),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def _calls(agent):
    return agent.client.chat.completions.create.call_args_list


def _last_user(call):
    return [m for m in call.kwargs["messages"] if m.get("role") == "user"][-1]["content"]


class TestFullSlotTruncation:
    def test_wrong_window_is_corrected_and_the_call_recovers_once(self, agent):
        agent.context_compressor.update_model("test/model", 262144, base_url=agent.base_url, provider="custom")
        agent.client.chat.completions.create.side_effect = [
            _partial(usage=FULL_SLOT), _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        executed: list = []
        result = _run(agent, executed=executed)

        assert result["final_response"] == "Done." and result["completed"] is True
        assert len(executed) == 1, "the truncated call must never run, the recovered one runs exactly once"
        assert '"content":"x = 1"' in executed[0][1] or executed[0][1].get("content") == "x = 1"
        assert agent.context_compressor.context_length == 65536, "learned from what the full slot proved"
        retry = _calls(agent)[1]
        assert NUDGE_MARK in _last_user(retry) and "smaller" in _last_user(retry)
        # Only 2,413 tokens remain in the slot, so no larger cap can honestly be offered: none is sent
        # (the server's own limit applies) rather than one the window cannot hold.
        assert retry.kwargs.get("max_tokens") is None or retry.kwargs["max_tokens"] <= 65536 - 63123
        stored = [m for m in result["messages"] if m.get("role") == "user"]
        assert [m["content"] for m in stored] == ["write big.py"], "the nudge is request-only"
        assert not any("c_bad" in str(m.get("tool_calls", "")) for m in result["messages"]), "truncated call not stored"

    def test_server_window_wins_over_the_inferred_one(self, agent):
        agent.context_compressor.update_model("test/model", 262144, base_url=agent.base_url, provider="custom")
        agent.client.chat.completions.create.side_effect = [
            _partial(usage=FULL_SLOT), _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        with (
            patch("model_tools.handle_function_call", return_value='{"success": true}'),
            patch("agent.turn_tool_call_recovery._server_window", return_value=131072),
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("write big.py")
        assert agent.context_compressor.context_length == 131072

    def test_exhausted_ladder_names_the_context_window_not_the_output_limit(self, agent):
        agent.context_compressor.update_model("test/model", 262144, base_url=agent.base_url, provider="custom")
        agent.client.chat.completions.create.side_effect = [_partial(usage=FULL_SLOT) for _ in range(12)]
        executed: list = []
        result = _run(agent, executed=executed)

        assert executed == []
        assert len(_calls(agent)) == 7  # first attempt + 6 distinct recovery attempts, not an endless loop
        text = result["final_response"]
        assert "65,536-token context window" in text and "/compress" in text and "output limit, not" not in text.lower() or "context limit, not an output limit" in text
        assert result["partial"] is True and result["failure_reason"] == "truncated"
        assert agent._tool_trunc_attempts == 0, "state is cleared for the next turn"


class TestOutputCapTruncation:
    def test_cap_doubles_and_the_nudge_arrives_from_the_second_attempt(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _partial(usage=CAP_HIT), _partial(usage={**CAP_HIT, "completion_tokens": 8192}),
            _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        executed: list = []
        result = _run(agent, executed=executed)

        assert result["completed"] is True and len(executed) == 1
        _first, second, third = _calls(agent)[:3]
        assert second.kwargs["max_tokens"] > 4096 and third.kwargs["max_tokens"] > second.kwargs["max_tokens"]
        assert NUDGE_MARK in _last_user(third)

    def test_a_known_model_limit_bounds_the_cap(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _partial(usage=CAP_HIT), _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        with patch("agent.turn_truncation._model_output_limit", return_value=6000):
            result = _run(agent)
        assert result["completed"] is True
        assert _calls(agent)[1].kwargs["max_tokens"] <= 6000

    def test_exhausted_ladder_names_the_output_limit(self, agent):
        agent.client.chat.completions.create.side_effect = [_partial(usage=CAP_HIT) for _ in range(12)]
        result = _run(agent)
        assert "hit its output limit on every one" in result["final_response"]
        assert "/model" in result["final_response"]


class TestSequentialCalls:
    def test_progress_between_truncations_resets_the_ladder(self, agent):
        """Two separate truncations in one long turn each get a full ladder: a tool batch that ran
        in between is progress, so neither 'uses up' the other's attempts."""
        responses = [_whole("c1")]
        for _ in range(4):
            responses.append(_partial(usage=CAP_HIT))
        responses += [_whole("c2")]
        for _ in range(4):
            responses.append(_partial(usage=CAP_HIT))
        responses += [_whole("c3"), _mock_response(content="All three written.", finish_reason="stop")]
        agent.client.chat.completions.create.side_effect = responses
        executed: list = []
        result = _run(agent, executed=executed)

        assert result["final_response"] == "All three written." and result["completed"] is True
        assert len(executed) == 3
        assert len(_calls(agent)) == len(responses)

    def test_completed_tool_results_survive_a_later_truncation(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _whole("c1", "a.py"), _partial(usage=CAP_HIT), _whole("c2", "b.py"),
            _mock_response(content="Both done.", finish_reason="stop"),
        ]
        result = _run(agent)
        tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_rows) == 2 and result["completed"] is True


class TestRoutersThatRewriteFinishReason:
    def test_truncated_args_with_finish_reason_tool_calls_recover_too(self, agent):
        """The tool-validation path (router turned ``length`` into ``tool_calls``) used to end the turn
        on the first sighting; it now takes the same ladder."""
        agent.client.chat.completions.create.side_effect = [
            _partial(finish="tool_calls"), _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        executed: list = []
        result = _run(agent, executed=executed)
        assert result["completed"] is True and len(executed) == 1
        assert NUDGE_MARK in _last_user(_calls(agent)[1])

    def test_still_truncated_after_the_ladder_ends_the_turn_without_running_anything(self, agent):
        agent.client.chat.completions.create.side_effect = [_partial(finish="tool_calls") for _ in range(12)]
        executed: list = []
        result = _run(agent, executed=executed)
        assert executed == [] and result["partial"] is True
        assert "could not be completed" in result["final_response"]


class TestDroppedStreams:
    def test_temporary_disconnects_recover_without_touching_the_window(self, agent):
        agent.context_compressor.update_model("test/model", 262144, base_url=agent.base_url, provider="custom")
        stalls = []
        for _ in range(2):
            stub = _partial()
            stub.id = PARTIAL_STREAM_STUB_ID
            stalls.append(stub)
        agent.client.chat.completions.create.side_effect = stalls + [
            _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        executed: list = []
        result = _run(agent, executed=executed)
        assert result["completed"] is True and len(executed) == 1
        assert agent.context_compressor.context_length == 262144, "a network drop says nothing about the window"


class TestRegression:
    def test_a_plain_text_truncation_still_continues(self, agent):
        """The text path is untouched: partial text is kept and the model is asked to carry on."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Part one, ", finish_reason="length"),
            _mock_response(content="part two.", finish_reason="stop"),
        ]
        result = _run(agent)
        assert result["completed"] is True and "Part one" in result["final_response"] and "part two." in result["final_response"]


class TestLongSessionsAndModelSwitches:
    def test_a_long_coding_session_survives_repeated_truncations_without_repeating_work(self, agent):
        """40 sequential file writes with a truncation every fifth call: every file is written exactly once, in
        order, no truncated call ever runs, and no 'continue' is needed."""
        responses, expected = [], []
        for i in range(40):
            if i % 5 == 4:
                responses.append(_partial(call_id=f"trunc{i}", usage=CAP_HIT))
            responses.append(_whole(f"c{i}", f"f{i}.py"))
            expected.append(f"f{i}.py")
        responses.append(_mock_response(content="All forty files written.", finish_reason="stop"))
        agent.client.chat.completions.create.side_effect = responses
        executed: list = []
        result = _run(agent, executed=executed)

        assert result["completed"] is True and result["final_response"] == "All forty files written."
        import json as _json
        paths = [(_json.loads(a) if isinstance(a, str) else a)["path"] for _n, a in executed]
        assert paths == expected, "each file written once, in order"
        tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_rows) == 40, "every completed result is still in the session"

    @pytest.mark.parametrize("limit", [6000, 16000, 64000, None])
    def test_the_cap_follows_whichever_model_is_active(self, agent, limit):
        """Switching between models with different output limits: each gets its own ceiling, never one assumed value."""
        agent.client.chat.completions.create.side_effect = [
            _partial(usage=CAP_HIT), _whole("c_ok"), _mock_response(content="Done.", finish_reason="stop"),
        ]
        with patch("agent.turn_truncation._model_output_limit", return_value=limit):
            result = _run(agent)
        assert result["completed"] is True
        cap = _calls(agent)[1].kwargs.get("max_tokens")
        if limit is None:
            assert cap and cap > 4096
        else:
            assert cap is None or cap <= limit
