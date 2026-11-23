"""A reasoning model that spends the whole output budget on reasoning (finish_reason="length", no
visible text) must escalate a bounded number of times instead of repeating the same request:

  retry 1: reasoning off for one request + a larger output cap (when one can be sized honestly)
  retry 2: the same + a request-only "answer now" nudge
  then:    one message naming what was tried; the user's message stays in the session.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH

NUDGE_MARK = "Stop deliberating"


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _reasoning_only_response(completion_tokens=None):
    from tests.agent.test_run_agent import _mock_assistant_msg

    usage = None
    if completion_tokens:
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=completion_tokens, total_tokens=100 + completion_tokens,
        )
    return SimpleNamespace(
        id="chatcmpl-reasoning-only", model="test/model",
        choices=[SimpleNamespace(
            index=0, message=_mock_assistant_msg(content="", reasoning="step after step of deliberation"),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=usage,
    )


def _answer(text="Here is the answer."):
    from tests.agent.test_run_agent import _mock_response

    return _mock_response(content=text, finish_reason="stop")


def _run(agent, message="what is the answer?"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def _last_user_text(call):
    users = [m for m in call.kwargs["messages"] if m.get("role") == "user"]
    return users[-1]["content"]


def _calls(agent):
    return agent.client.chat.completions.create.call_args_list


def _reasoning_on_wire(call):
    return (call.kwargs.get("extra_body") or {}).get("reasoning")


class TestEscalation:
    def test_first_retry_turns_reasoning_off_and_raises_the_cap(self, loop_agent):
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent._supports_reasoning_extra_body = lambda: True
        loop_agent.client.chat.completions.create.side_effect = [_reasoning_only_response(4096), _answer()]
        result = _run(loop_agent)

        assert result["completed"] is True and "the answer" in result["final_response"]
        first, second = _calls(loop_agent)
        assert _reasoning_on_wire(first) == {"enabled": True, "effort": "high"}
        assert _reasoning_on_wire(second) == {"enabled": False, "effort": "none"}
        assert second.kwargs.get("max_tokens") == 8192  # sized from the 4096 tokens the server produced
        assert NUDGE_MARK not in _last_user_text(second), "rung 1 changes parameters only"
        assert loop_agent._ephemeral_reasoning_off is False

    def test_no_cap_is_guessed_when_the_server_ceiling_is_unknown(self, loop_agent):
        """No max_tokens sent and no usage reported: an invented cap could LOWER the server's own default."""
        loop_agent.client.chat.completions.create.side_effect = [_reasoning_only_response(None), _answer()]
        _run(loop_agent)
        second = _calls(loop_agent)[1]
        assert second.kwargs.get("max_tokens") is None

    def test_second_retry_adds_a_request_only_nudge(self, loop_agent):
        loop_agent.client.chat.completions.create.side_effect = [
            _reasoning_only_response(4096), _reasoning_only_response(8192), _answer("Forty-two."),
        ]
        result = _run(loop_agent)

        assert result["completed"] is True and result["final_response"] == "Forty-two."
        first, second, third = _calls(loop_agent)
        assert NUDGE_MARK not in _last_user_text(second)
        assert _last_user_text(third).startswith("what is the answer?")
        assert NUDGE_MARK in _last_user_text(third)
        assert third.kwargs["max_tokens"] > second.kwargs["max_tokens"]
        # The nudge never reaches the stored transcript, and the user's turn is not rewritten.
        stored_users = [m["content"] for m in result["messages"] if m.get("role") == "user"]
        assert stored_users == ["what is the answer?"]
        assert loop_agent._request_only_nudge is None

    def test_default_is_two_extra_attempts_then_one_clear_message(self, loop_agent):
        loop_agent.client.chat.completions.create.side_effect = [_reasoning_only_response(4096) for _ in range(6)]
        result = _run(loop_agent)

        assert len(_calls(loop_agent)) == 3  # first attempt + 2 escalations, not 5 identical ones
        assert result["completed"] is False and result["partial"] is True
        text = result["final_response"]
        assert "No answer was produced" in text and "3 attempts" in text
        assert "reasoning switched off" in text and "told to answer now" in text
        assert "/reasoning none" in text and "/retry" in text
        assert result["failure_reason"] == "truncated" and result["failure_retryable"] is True
        # /retry needs the user's message in the session; no synthetic rows may remain.
        users = [m for m in result["messages"] if m.get("role") == "user"]
        assert [m["content"] for m in users] == ["what is the answer?"]
        assert not any(m.get("_length_continuation_nudge") for m in result["messages"])
        assert not any(
            m.get("role") == "assistant" and not (m.get("content") or "").strip() and not m.get("tool_calls")
            for m in result["messages"]
        )
        assert loop_agent._ephemeral_reasoning_off is False and loop_agent._len_reasoning_streak == 0

    def test_attempt_limit_is_configurable(self, loop_agent, monkeypatch):
        monkeypatch.setenv("HERMES_REASONING_TRUNCATION_RETRIES", "1")
        loop_agent.client.chat.completions.create.side_effect = [_reasoning_only_response(4096) for _ in range(6)]
        result = _run(loop_agent)
        assert len(_calls(loop_agent)) == 2
        assert "2 attempts" in result["final_response"]

    def test_inline_think_only_content_takes_the_same_ladder(self, loop_agent):
        from tests.agent.test_run_agent import _mock_assistant_msg

        inline = _reasoning_only_response(4096)
        inline.choices[0].message = _mock_assistant_msg(content="<think>long deliberation, no answer")
        loop_agent.client.chat.completions.create.side_effect = [inline, _answer()]
        result = _run(loop_agent)
        assert result["completed"] is True and len(_calls(loop_agent)) == 2

    def test_streak_does_not_carry_into_the_next_turn(self, loop_agent):
        loop_agent.client.chat.completions.create.side_effect = [
            _reasoning_only_response(4096), _answer(),
            _reasoning_only_response(4096), _answer("second turn"),
        ]
        _run(loop_agent)
        result = _run(loop_agent, "and another question")
        assert result["final_response"] == "second turn"
        third = _calls(loop_agent)[3]
        assert NUDGE_MARK not in _last_user_text(third)  # a fresh turn starts at rung 1 again

    def test_visible_text_truncation_keeps_the_ordinary_continuation(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="part one. ", finish_reason=FINISH_REASON_LENGTH), _answer("part two."),
        ]
        result = _run(loop_agent)
        assert result["completed"] is True and "part one." in result["final_response"]
        assert NUDGE_MARK not in _last_user_text(_calls(loop_agent)[1])


class TestLocalServerThinkingSwitch:
    """llama.cpp-style servers ignore reasoning_effort; their template reads chat_template_kwargs."""

    @staticmethod
    def _agent(base_url, override=True):
        return SimpleNamespace(
            base_url=base_url, provider="custom", api_mode="chat_completions", _wire_ephemeral_reasoning_off=override,
        )

    def test_local_endpoint_gets_enable_thinking_false_on_the_override_request(self):
        from agent.chat_completion_helpers import _template_thinking_off

        out = _template_thinking_off(self._agent("http://192.168.20.193:8080/v1"), {"model": "m", "extra_body": {"a": 1}})
        assert out["extra_body"] == {"a": 1, "chat_template_kwargs": {"enable_thinking": False}}

    def test_hosted_endpoint_and_normal_requests_are_untouched(self):
        from agent.chat_completion_helpers import _template_thinking_off

        assert "extra_body" not in _template_thinking_off(self._agent("https://api.example.com/v1"), {"model": "m"})
        assert "extra_body" not in _template_thinking_off(
            self._agent("http://192.168.20.193:8080/v1", override=False), {"model": "m"}
        )
