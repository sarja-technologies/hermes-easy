"""Truncation recovery (``finish_reason == "length"``) for the conversation turn loop.

Handles thinking-budget exhaustion, repetition-dominated truncation, content-filter stream
stalls escalated to the fallback chain, text continuation nudges (up to 4, with the ceiling
exit that drops the fragment trail), truncated tool-call retries with max_tokens boosts, and
the final roll-back. Nothing here imports ``agent.conversation_loop`` at module level
(cycle); loop-internal helpers are imported lazily so tests patching them keep working.
"""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.error_classifier import FailoverReason
from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.repetition_guard import is_repetition_dominated
from agent.turn_api_call import stop_thinking_spinner
from agent.turn_failure_copy import content_policy_copy, provider_label_for, site_copy, stamp_failure
from agent.turn_retry_state import TurnRetryState
from agent.usage_pricing import normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from utils import env_int

logger = logging.getLogger("agent.conversation_loop")

# codex_responses only reaches ``finish_reason == "length"`` for a tool call cut off by
# max_output_tokens (turn_response_check.py::_derive_finish_reason); text truncation stays on
# the Codex incomplete continuation, so the text branch below never double-continues it.
_CONTINUABLE_MODES = {"chat_completions", "bedrock_converse", "anthropic_messages", "codex_responses"}
_THINK_TAG_RE = re.compile(r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>', re.IGNORECASE)
_TRUNCATED_FINAL = site_copy("truncated")
_FIRST_TRUNCATED_FINAL = _TRUNCATED_FINAL
# #106260: a stream that died on a context-overflow error after partial delivery must not seed a
# continuation — the transcript already cannot fit, and appending the partial stub grows every
# later request into the same overflow. End the turn via the recovery contract instead.
_CONTEXT_OVERFLOW_PARTIAL_FINAL = (
    "The request no longer fits the model's context window, so the partial "
    "response was not continued. Continue in a fresh session (/new; gateway "
    "chats are reset automatically)."
)

def collapse_continuation_trail(
    agent: Any, messages: list[dict[str, Any]], current_turn_user_idx: Any, *,
    finish_reason: str, parts: Optional[list[str]] = None,
) -> str:
    """Drop this turn's ``_length_continuation_fragment``/``_nudge`` rows and append one
    assistant row holding the joined, think-stripped partial (a held turn's rewrite); returns
    that text ("" none).

    ``parts=None`` (retry exhaustion, #119001): the text comes from the fragment rows and
    nothing happens without a valid turn index or a trail — an unanswered synthetic nudge
    must never be persisted, and an earlier turn's rows must never be read. Explicit
    ``parts`` (the continuation ceiling) always appends, scanning from 0 without an index.
    """
    idx = current_turn_user_idx
    valid_idx = isinstance(idx, int) and idx >= 0
    if parts is None and not (valid_idx and idx < len(messages)):
        return ""
    turn_start = idx + 1 if valid_idx else 0
    fragment_parts: list[str] = []
    retained: list[Any] = []
    found_trail = False
    for message in messages[turn_start:]:
        if isinstance(message, dict) and (
            message.get("_length_continuation_fragment") or message.get("_length_continuation_nudge")
        ):
            found_trail = True
            content = message.get("content")
            if message.get("_length_continuation_fragment") and isinstance(content, str) and content:
                fragment_parts.append(content)
            continue
        retained.append(message)
    if parts is None and not found_trail:
        return ""
    messages[turn_start:] = retained
    from agent.conversation_loop import _join_truncated_parts
    raw = fragment_parts if parts is None else parts
    join_parts = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            join_parts.append(item)
        elif isinstance(item, str) and item:
            join_parts.append((item, False))
    partial = agent._strip_think_blocks(
        _join_truncated_parts(join_parts)
    ).strip()
    if partial and getattr(agent, "_hold_stream", False) is True:
        # A held turn showed none of these fragments: the stitched text reaches the user and the
        # stored row only through the turn's transform snapshot (fail closed: the withheld notice).
        from agent.turn_finalizer import apply_llm_output_transform
        partial, _, _ = apply_llm_output_transform(agent, partial, turn_id=getattr(agent, "_current_turn_id", "") or "")
    if partial:
        append_message(messages, {"role": "assistant", "content": partial, "finish_reason": finish_reason})
    agent._session_messages = messages
    return partial


_THINKING_EXHAUSTED = (
    "💭 Reasoning exhausted the output token budget — no visible response was produced.",
    "⚠️ **Thinking Budget Exhausted**\n\nThe model used all its output tokens on reasoning "
    "and had none left for the actual response.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning minimal`\n"
    "→ Or switch to a larger/non-reasoning model with `/model`",
    "Model used all output tokens on reasoning with none left "
    "for the response. Try lowering reasoning effort or increasing max_tokens.",
)

def repetition_copy(stopping: str, outcome: str, refusal: str) -> tuple[str, str, str]:
    """(log line, user copy, error) for a repetition-dominated abort; only the clauses naming
    where the turn stopped differ between the length path and the stop path."""
    return (
        f"🔁 Response dominated by repeated text — stopping {stopping}.",
        "⚠️ **Response Stopped — Repetition Detected**\n\nThe model fell into a repetition loop while "
        f"writing this response, {outcome}\n\n→ Switch to a different model with `/model`\n"
        "→ Or resend your message (your conversation history is preserved)",
        f"Model output entered a repetition loop{refusal} degenerate response.",
    )


_REPETITION_DOMINATED = repetition_copy(
    "instead of continuing a degenerate response",
    "so continuing would only produce more repeated text. The partial response was discarded.",
    " and was truncated mid-loop; refusing to continue a",
)
_REPETITION_STREAM_CUT = repetition_copy(
    "the stream mid-loop",
    "so the stream was stopped instead of running on. The partial response was discarded.",
    " and the stream was cut mid-loop; discarding the",
)
_CEILING_NO_TEXT = (
    "⚠️ **No visible answer was produced.** The model hit its output-token limit on every "
    "continuation attempt — its reasoning consumed the entire budget each time.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning none`\n→ Or raise max_tokens for this model"
)
# Below this many free tokens the prompt itself filled the window: a continuation nudge +
# fragment costs ~100 tokens per attempt, so retrying only shrinks the room (#106120).
_MIN_CONTINUATION_HEADROOM = 512
# A long answer (a whole module, a long report) legitimately spans many output windows: keep going while
# every round adds real text, and stop at the cap or at the first round that adds nothing.
_MAX_TEXT_CONTINUATIONS = 12
_FREE_CONTINUATIONS = 4  # rounds 1-3 never need to prove progress (thinking-only rounds are normal)
_SUBSTANTIVE_CHARS = 200  # a round past the free ones must have written at least this much to earn another
_WINDOW_FILLED = (
    "⚠️ **Context window full.** The prompt used {prompt:,} of this model's {ctx:,}-token "
    "context window, leaving no room to answer in. This is a context-window limit, not an "
    "output-length limit.\n\nTo fix this:\n→ Compress the conversation with `/compress` or start "
    "a new session\n→ Or raise the model's context window (e.g. Ollama `num_ctx`)"
)


def _prompt_filled_window(agent: Any, response: Any) -> Optional[tuple[int, int]]:
    """``(prompt_tokens, context_length)`` when this response's usage shows the prompt left
    less than ``_MIN_CONTINUATION_HEADROOM`` in the window compression resolves for the
    model; ``None`` (keep continuing) when either number is unknown."""
    ctx = int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0)
    usage = getattr(response, "usage", None)
    if not (ctx and usage):
        return None
    prompt = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode).prompt_tokens
    return (prompt, ctx) if prompt and ctx - prompt < _MIN_CONTINUATION_HEADROOM else None


def normalize_response_for_agent(agent: Any, response: Any) -> Any:
    """One OpenAI-style message from any transport; Anthropic strips the OAuth tool prefix."""
    if agent.api_mode == "anthropic_messages":
        return agent._get_transport().normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    return agent._get_transport().normalize_response(response)


def partial_result(
    messages: list[dict[str, Any]], api_call_count: int, final_response: str,
    error: Optional[str] = None, *, failed: bool = False, compression_exhausted: bool = False,
) -> dict[str, Any]:
    """Typed incomplete-turn result (``partial`` unless ``failed``); ``error`` defaults to
    ``final_response``. ``compression_exhausted`` carries the #98722 typed bit the gateway
    consumes to reset/move future input to a clean session (see run_turn.py)."""
    result = {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        ("failed" if failed else "partial"): True,
        "error": final_response if error is None else error,
    }
    if compression_exhausted:
        result["compression_exhausted"] = True
    return result


@dataclass
class TruncationVerdict:
    """Outcome of ``recover_from_truncation``.

    ``action``: ``"return"`` (end the turn with ``result``), ``"break"`` (a
    ``_retry.restart_with_*`` flag is set — restart the API call), ``"continue"``
    (re-issue the same call immediately) or ``"fallthrough"`` (unreachable in practice:
    every path exits, kept for the contract). The remaining fields are the loop locals
    the handler may have rebound."""

    action: str
    result: Optional[dict[str, Any]]
    messages: list[dict[str, Any]]
    length_continue_retries: int
    truncated_response_parts: list[tuple[str, bool]]
    truncated_tool_call_retries: int
    retry_count: int
    compression_attempts: int


@dataclass(kw_only=True)
class _Trunc(TruncationVerdict):
    """Working state for the truncation phases — the verdict itself, plus the read-only
    call context; phases mutate the loop-local fields and ``done()`` stamps the action."""

    agent: Any
    response: Any
    finish_reason: str
    conversation_history: Any
    api_call_count: int
    effective_task_id: Any
    current_turn_user_idx: Any
    action: str = "fallthrough"
    result: Optional[dict[str, Any]] = None
    window_filled: Optional[tuple[int, int]] = None  # (prompt_tokens, context_length)
    api_kwargs: Any = None  # the request that was just truncated

    def done(self, action: str, result: Optional[dict[str, Any]] = None) -> TruncationVerdict:
        self.action, self.result = action, result
        return self

    def end_turn(
        self, final_response: str, error: Optional[str] = None, *,
        result_messages: Optional[list[dict[str, Any]]] = None, cleanup: bool = True,
        failed: bool = False, compression_exhausted: bool = False,
        failure: tuple[str, bool] = ("truncated", True),
    ) -> TruncationVerdict:
        """Persist and end the turn as partial (or ``failed``).

        ``compression_exhausted`` forwards the #98722 typed bit so the gateway can
        move future input off a bloated session (run_turn.py consumes it). ``failure`` is
        the ``(failure_reason, retryable)`` verdict for the UI descriptor.
        """
        agent = self.agent
        if cleanup:
            agent._cleanup_task_resources(self.effective_task_id)
        agent._persist_session(self.messages, self.conversation_history)
        return self.done("return", stamp_failure(partial_result(
            self.messages if result_messages is None else result_messages, self.api_call_count,
            final_response, error, failed=failed, compression_exhausted=compression_exhausted,
        ), *failure))

    @property
    def is_stub(self) -> bool:
        return getattr(self.response, "id", "") == PARTIAL_STREAM_STUB_ID


def _abort_reason(agent: Any, content: Any, has_tool_calls: bool) -> Optional[tuple]:
    """``(vprint, user response, error)`` when continuation must NOT be attempted:
    thinking exhausted the budget (reasoning blocks with no visible text after them —
    ``content=None`` from non-<think> models is normal truncation), or a repetition loop
    burned the budget on one fragment (reasoning stripped first)."""
    if has_tool_calls:
        return None
    if content and _THINK_TAG_RE.search(content) and not agent._has_content_after_think_block(content):
        return _THINKING_EXHAUSTED
    visible = agent._strip_think_blocks(content) if isinstance(content, str) else content
    if visible and is_repetition_dominated(visible):
        return _REPETITION_DOMINATED
    return None


def _content_filter_fallback(st: _Trunc, _retry: TurnRetryState) -> Optional[TruncationVerdict]:
    """Content-filter stream stall → fallback. ``_content_filter_terminated`` is
    content-deterministic, so escalate before retrying the primary; without a fallback
    fall through to normal continuation (best-effort, may loop)."""
    agent = st.agent
    if not (
        getattr(st.response, "_content_filter_terminated", False)
        and agent._fallback_index < len(agent._fallback_chain)
    ):
        return None
    agent._vprint(
        f"{agent.log_prefix}🛡️  Content filter terminated stream — activating fallback provider...",
        force=True, diagnostic=True,
    )
    agent._emit_diagnostic_status("Content filter terminated stream; switching to fallback...")
    if agent._try_activate_fallback(reason=FailoverReason.content_policy_blocked):
        logger.warning("%sContent filter terminated the stream — switched to fallback", agent.log_prefix)
        # Roll partial content back to the last clean turn so the fallback gets a
        # coherent continuation point; unmark survivors (their text left the partial).
        if st.truncated_response_parts:
            st.messages = agent._get_messages_up_to_last_assistant(st.messages)
        for _frag in st.messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)
        agent._session_messages = st.messages
        st.length_continue_retries = 0
        st.truncated_response_parts = []
        st.retry_count = 0
        st.compression_attempts = 0
        _retry.primary_recovery_attempted = False
        _retry.restart_with_rebuilt_messages = True
        return st.done("break")
    agent._vprint(
        f"{agent.log_prefix}⚠️  No fallback provider configured — retrying with same provider "
        f"(may re-hit filter)...",
        force=True, diagnostic=True,
    )
    return None


def _continue_text(st: _Trunc, _retry: TurnRetryState, assistant_message: Any) -> TruncationVerdict:
    """Text truncation (no tool calls): append the fragment + a continuation nudge (up to
    4), then the ceiling exit that drops the fragment trail and keeps the stitched partial.
    Never appends an interim assistant row with NO visible content — strict providers
    reject it with 400 — only the nudge."""
    from agent.conversation_loop import _get_continuation_prompt, _join_truncated_parts

    agent = st.agent
    messages = st.messages
    st.length_continue_retries += 1
    n = st.length_continue_retries
    _interim_content = getattr(assistant_message, "content", None)
    if not _interim_content and not st.is_stub:
        # Thinking-only truncation: continuing with thinking ON re-burns the budget.
        agent._ephemeral_reasoning_off = True
    if _interim_content:
        reset_reasoning_only_ladder(agent)  # this turn wrote text: the reasoning-only chain is over
        interim_msg = agent._build_assistant_message(assistant_message, st.finish_reason)
        interim_msg["_length_continuation_fragment"] = True  # ceiling exit drops these
        append_message(messages, interim_msg)
        st.truncated_response_parts.append((_interim_content, st.is_stub))

    filled = st.window_filled
    progressing = n < _FREE_CONTINUATIONS or len((_interim_content or "").strip()) >= _SUBSTANTIVE_CHARS
    if n < _MAX_TEXT_CONTINUATIONS and progressing and filled is None:
        _dropped_tools = getattr(st.response, "_dropped_tool_names", None)
        if st.is_stub and _dropped_tools:
            agent._vprint(
                f"{agent.log_prefix}↻ Stream interrupted mid "
                f"tool-call ({', '.join(_dropped_tools[:3])}) — requesting chunked retry ({n}/{_MAX_TEXT_CONTINUATIONS})...", diagnostic=True,
            )
        elif st.is_stub:
            agent._vprint(f"{agent.log_prefix}↻ Stream interrupted — requesting continuation ({n}/{_MAX_TEXT_CONTINUATIONS})...", diagnostic=True)
        else:
            agent._vprint(f"{agent.log_prefix}↻ Requesting continuation ({n}/{_MAX_TEXT_CONTINUATIONS})...", diagnostic=True)
        append_message(messages, {
            "role": "user", "content": _get_continuation_prompt(st.is_stub, _dropped_tools),
            "_length_continuation_nudge": True,
        })
        agent._clear_status_buffer()
        with suppress(Exception):
            agent._emit_diagnostic_wait(f"↻ reply hit the output limit — continuing ({n}/{_MAX_TEXT_CONTINUATIONS})")
        agent._session_messages = messages
        _retry.restart_with_length_continuation = True
        return st.done("break")

    # Unanswered continue nudges made every later turn re-truncate: drop the trail.
    partial_response = collapse_continuation_trail(
        agent, messages, st.current_turn_user_idx, finish_reason="length",
        parts=st.truncated_response_parts,
    )
    # The one-shot reasoning-off override must not leak into the next turn.
    agent._ephemeral_reasoning_off = False
    agent._vprint(
        f"{agent.log_prefix}⚠️  Not continuing — each attempt would only grow the prompt."
        if filled is not None else
        f"{agent.log_prefix}⚠️  Response still truncated after {n} continuation attempt(s) — "
        + ("keeping the partial response received so far." if partial_response
           else "no visible text was produced."),
        force=True, diagnostic=True,
    )
    if filled is not None:
        notice = _WINDOW_FILLED.format(prompt=filled[0], ctx=filled[1])
        return st.end_turn(
            f"{partial_response}\n\n{notice}" if partial_response else notice,
            f"Prompt used {filled[0]} of {filled[1]} context tokens; no room to answer",
        )
    return st.end_turn(
        partial_response or _CEILING_NO_TEXT,
        f"Response remained truncated after {n} continuation attempts",
    )


# --- reasoning-only truncation ladder -------------------------------------------------------------
# A reasoning model can spend the whole output budget on reasoning and finish with
# ``finish_reason="length"`` and no visible text. Repeating the same request re-burns the budget
# (minutes per attempt on a local model), so the turn escalates a bounded number of times:
#   1. reasoning off for one request and a larger output cap (what the provider honours),
#   2. the same plus a request-only system nudge to stop deliberating and answer from the reasoning so far,
# then ends with one message naming what was tried. The user's message stays in the session.
_REASONING_ONLY_NUDGE = (
    "[System: Your previous attempt used its entire output budget on internal reasoning and never "
    "produced a visible answer. Stop deliberating. Answer now, directly, using the reasoning you have "
    "already done; if something is still uncertain, give your best answer and say what is uncertain.]"
)
REASONING_RETRIES_ENV = "HERMES_REASONING_TRUNCATION_RETRIES"
_DEFAULT_REASONING_RETRIES = 2
_CAP_HEADROOM = 256  # tokens kept free below the context window when raising the output cap


def reasoning_retry_limit() -> int:
    """Extra attempts after a reasoning-only truncation (``HERMES_REASONING_TRUNCATION_RETRIES``, default 2, max 4)."""
    return max(0, min(env_int(REASONING_RETRIES_ENV, _DEFAULT_REASONING_RETRIES), 4))


def reset_reasoning_only_ladder(agent: Any) -> None:
    """Forget the ladder's progress (turn start, a turn that produced text, the final stop)."""
    agent._len_reasoning_streak = 0
    agent._len_reasoning_tried = []
    agent._len_reasoning_last_call = -1
    agent._length_escalation = None
    agent._request_only_nudge = None


def _reasoning_text_of(agent: Any, message: Any, content: Any) -> str:
    """The reasoning a response carried (structured field, else inline think blocks); never shown to the user."""
    text: Any = ""
    with suppress(Exception):
        text = agent._extract_reasoning(message) or ""
    if not (isinstance(text, str) and text.strip()) and isinstance(content, str) and _THINK_TAG_RE.search(content):
        text = content
    return text.strip() if isinstance(text, str) else ""


def _escalated_output_cap(st: _Trunc) -> Optional[int]:
    """A larger output cap for the next attempt, or ``None`` when none can honestly be offered.

    Sized from what the request really allowed (the cap it carried, else the tokens the server
    produced before cutting it off) and held below the context window; never a cap at or under
    what just failed, and never a guess when neither number is known (a made-up 8K could LOWER a
    server's own default)."""
    agent = st.agent
    requested = agent._requested_output_cap_from_api_kwargs(st.api_kwargs)
    usage = getattr(st.response, "usage", None)
    observed = prompt = 0
    if usage:
        with suppress(Exception):
            canonical = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode)
            observed, prompt = int(canonical.output_tokens or 0), int(canonical.prompt_tokens or 0)
    if not (requested or observed):
        return None
    configured = agent.max_tokens if isinstance(agent.max_tokens, int) else 0
    base = max(requested or 0, configured, observed, 4096)
    cap = boosted_output_cap(agent, requested, 1, base=base)
    ctx = int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0)
    if ctx and prompt:
        cap = min(cap, ctx - prompt - _CAP_HEADROOM)
    return cap if cap > max(requested or 0, observed) else None


def _escalate_reasoning_only(
    st: _Trunc, _retry: TurnRetryState, message: Any, content: Any,
) -> Optional[TruncationVerdict]:
    """Handle a length truncation that carried reasoning but no visible text; ``None`` for any other
    truncation (visible text, no reasoning at all), which keeps the ordinary continuation path."""
    agent = st.agent
    visible = agent._strip_think_blocks(content).strip() if isinstance(content, str) else ""
    if visible or not _reasoning_text_of(agent, message, content):
        return None

    last_call, streak = getattr(agent, "_len_reasoning_last_call", -1), getattr(agent, "_len_reasoning_streak", 0)
    consecutive = isinstance(last_call, int) and isinstance(streak, int) and 0 <= st.api_call_count - last_call <= 1
    streak = streak + 1 if consecutive else 1
    agent._len_reasoning_streak, agent._len_reasoning_last_call = streak, st.api_call_count
    tried = getattr(agent, "_len_reasoning_tried", None)
    tried = list(tried) if consecutive and isinstance(tried, list) else []

    if streak <= reasoning_retry_limit():
        steps = ["reasoning switched off"]
        agent._ephemeral_reasoning_off = True
        cap = _escalated_output_cap(st)
        agent._length_escalation = {"cap": cap}
        if cap:
            steps.append(f"output limit raised to {cap:,} tokens")
        if streak >= 2:
            # Request-only (see assemble_api_request): a stored row next to the unanswered user turn
            # would be merged into it by the sequence repair and persisted as a doubled message.
            agent._request_only_nudge = _REASONING_ONLY_NUDGE
            steps.append("told to answer now instead of deliberating")
        tried.append(", ".join(steps))
        agent._len_reasoning_tried = tried
        agent._session_messages = st.messages
        agent._vprint(
            f"{agent.log_prefix}↻ Reasoning used the whole output budget with no answer — retrying: {tried[-1]} "
            f"({streak}/{reasoning_retry_limit()})...", force=True, diagnostic=True,
        )
        with suppress(Exception):
            agent._emit_diagnostic_wait(f"↻ model reasoned without answering — retrying with {tried[-1]}")
        _retry.restart_with_length_continuation = True
        return st.done("break")

    # Out of attempts: one message, the user's own message stays in the session for /retry.
    partial = collapse_continuation_trail(
        agent, st.messages, st.current_turn_user_idx, finish_reason="length", parts=st.truncated_response_parts,
    )
    agent._ephemeral_reasoning_off = False
    reset_reasoning_only_ladder(agent)
    attempts = streak
    agent._vprint(
        f"{agent.log_prefix}⚠️  Reasoning used the whole output budget on all {attempts} attempt(s) — no answer produced.",
        force=True, diagnostic=True,
    )
    lines = ["1. First attempt: the output limit was reached while the model was still reasoning"]
    lines += [f"{i}. Retry {i - 1}: {step}" for i, step in enumerate(tried, start=2)]
    notice = (
        f"⚠️ **No answer was produced.** The model spent its whole output budget on reasoning in all "
        f"{attempts} attempt{'s' if attempts != 1 else ''} and never wrote a reply.\n\n"
        + "\n".join(lines)
        + "\n\nYour message is still in this session. Change one of these, then send `/retry`:\n"
        "→ `/reasoning none` (or `/reasoning low`) so it answers without long reasoning\n"
        "→ raise the output-token limit on the model server\n"
        "→ or switch model with `/model`"
    )
    return st.end_turn(
        f"{partial}\n\n{notice}" if partial else notice,
        f"Reasoning used the whole output budget on {attempts} attempt(s) with no visible answer",
    )


def _model_output_limit(agent: Any) -> Optional[int]:
    """The model's real max output tokens when Hermes knows it, else None."""
    if getattr(agent, "api_mode", None) != "anthropic_messages":
        return None
    # Local: only Anthropic-Messages turns need the adapter module.
    from agent.anthropic_adapter import _get_anthropic_max_output
    return _get_anthropic_max_output(getattr(agent, "model", None) or "")


def boosted_output_cap(agent: Any, requested_cap: Optional[int], n: int, base: Optional[int] = None) -> int:
    """Output budget for truncation retry ``n`` (1-based): ``base·2ⁿ``, never below the
    failed request's cap, at most ``max(32768, 2×cap)``, and never above the model's
    known output limit. ``base`` defaults to max_tokens, else the cap actually sent.

    A ceiling equal to the requested cap would re-send the same budget (#72770); a
    ceiling past the model limit only buys a provider 400 (#79715).
    """
    if base is None:
        base = agent.max_tokens or requested_cap or 4096
    anchor = requested_cap or base
    limit = _model_output_limit(agent)
    if limit and anchor >= limit:
        return anchor  # already at the model ceiling: doubling cannot help
    boost = min(max(base * (2 ** n), requested_cap or 0), max(32768, anchor * 2))
    return min(boost, limit) if limit else boost


def _retry_truncated_tool_call(st: _Trunc, _retry: TurnRetryState, api_kwargs: Any) -> TruncationVerdict:
    """Truncated tool call: never run it, never store it. ``plan_recovery`` classifies the stop (full
    context slot / output cap / dropped stream), corrects a wrong window, and arms a changed request
    (smaller pieces, bounded cap); the iteration restarts so preflight compression sees the corrected
    window. Only a spent ladder ends the turn, with copy naming the real cause."""
    from agent.turn_tool_call_recovery import MAX_ATTEMPTS, plan_recovery

    agent = st.agent
    plan = plan_recovery(agent, st.response, api_kwargs, stream_drop=st.is_stub)
    st.truncated_tool_call_retries += 1
    if plan.retry:
        what = "Stream interrupted mid tool-call" if st.is_stub else "Tool call cut off"
        summary = ", ".join(plan.steps) or "retrying"
        # The recovery is announced on the live status (a diagnostic wait notice), not as a scrollback line,
        # and the truncation banner buffered a moment ago is dropped: it is only shown if recovery fails.
        agent._clear_status_buffer()
        with suppress(Exception):
            agent._emit_diagnostic_wait(f"↻ {what.lower()}, not run — retrying: {summary}")
        agent._length_escalation = {"cap": plan.cap}
        _retry.restart_with_length_continuation = True
        return st.done("break")
    agent._flush_status_buffer()
    agent._vprint(
        f"{agent.log_prefix}⚠️  Tool call still truncated after {MAX_ATTEMPTS} recovery attempts — "
        "refusing to execute incomplete tool arguments.", force=True, diagnostic=True,
    )
    agent._cleanup_task_resources(st.effective_task_id)
    message, failure = plan.message, "truncated"
    if st.is_stub:  # keep the wording that tells a dropped network from a server that closed the stream cleanly
        clean_eof = getattr(st.response, "_clean_eof", False)
        message = site_copy("stream_closed_tool_call" if clean_eof else "stream_dropped_tool_call",
                            label=provider_label_for(agent.provider))
        failure = "truncated" if clean_eof else FailoverReason.timeout.value
    close_interrupted_tool_sequence(st.messages, message)
    return st.end_turn(message, plan.error, cleanup=False, failure=(failure, True))


def recover_from_truncation(
    agent: Any, response: Any, finish_reason: str, _retry: TurnRetryState, *,
    messages: list[dict[str, Any]], conversation_history: Any, api_kwargs: Any, api_call_count: int,
    effective_task_id: Any, current_turn_user_idx: Any, length_continue_retries: int,
    truncated_response_parts: list[tuple[str, bool]], truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int,
) -> TruncationVerdict:
    """Recover from a truncated response. Order is load-bearing: thinking exhaustion and
    repetition abort BEFORE any continuation; a content-filter stall escalates to the
    fallback chain BEFORE the primary is retried; text continuation (no tool calls) then
    truncated tool-call retry; finally roll back to the last complete assistant turn."""
    st = _Trunc(
        agent=agent, response=response, finish_reason=finish_reason,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id, current_turn_user_idx=current_turn_user_idx,
        messages=messages, length_continue_retries=length_continue_retries,
        truncated_response_parts=truncated_response_parts,
        truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
        compression_attempts=compression_attempts, api_kwargs=api_kwargs,
    )
    if getattr(response, "_runaway_repetition", False):
        # The streaming call cut a live repetition loop: a continuation would only re-enter it,
        # whatever the partial or its tool calls look like.
        line, user_response, error = _REPETITION_STREAM_CUT
        agent._vprint(f"{agent.log_prefix}{line}", force=True, diagnostic=True)
        return st.end_turn(user_response, error)
    st.window_filled = _prompt_filled_window(agent, response)
    if st.is_stub and getattr(response, "_clean_eof", False):
        _banner = ("Response truncated — server ended the stream without ever sending finish_reason "
                   "(no transport error — the server or a proxy closed the stream cleanly)")
    elif st.is_stub:
        _banner = "Response truncated — stream ended before completion"
    elif st.window_filled:
        _banner = (f"Response truncated (finish_reason='length') - the prompt filled the context window "
                   f"({st.window_filled[0]:,}/{st.window_filled[1]:,} tokens)")
    else:
        _banner = "Response truncated (finish_reason='length') - model hit max output tokens"
    # Buffered: a truncation the loop recovers from is a status, not an error, so the line is dropped when the
    # turn recovers and flushed (with the final message) only when every recovery attempt is exhausted.
    logger.warning("%s%s", agent.log_prefix, _banner)
    agent._buffer_vprint(f"⚠️  {_banner}")

    # #106260: a context-overflow error after partial delivery must not seed a
    # continuation. _partial_stream_stub marks such stubs _overflow_terminal and
    # leaves content empty; continuing would only re-send a larger request into
    # the same overflow. The stub path never raises, so this class never reached
    # recover_from_overflow's compress-and-retry on main either — ending the turn
    # replaces a growth loop, not a compression attempt.
    if getattr(st.response, "_overflow_terminal", False):
        agent._flush_status_buffer()
        agent._vprint(
            f"{agent.log_prefix}⚠️ Stream ended on a context-overflow error after "
            "partial delivery — not continuing (the request no longer fits the model's "
            "context window).",
            force=True, diagnostic=True,
        )
        # Prior tool batches can leave a tool-result tail; this path never reaches
        # finalize_turn (same as the truncated-tool-call terminal above).
        close_interrupted_tool_sequence(st.messages, _CONTEXT_OVERFLOW_PARTIAL_FINAL)
        # Carry the #98722 typed exhaustion bit so the gateway resets/moves future
        # input to a clean session instead of leaving this bloated one authoritative
        # for the next turn.
        return st.end_turn(
            _CONTEXT_OVERFLOW_PARTIAL_FINAL,
            error=_CONTEXT_OVERFLOW_PARTIAL_FINAL,
            failed=True,
            compression_exhausted=True,
            failure=("context_overflow", False),
        )

    _trunc_msg = normalize_response_for_agent(agent, response)
    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

    if (
        _trunc_msg is not None and not _trunc_has_tool_calls and not st.is_stub
        and st.window_filled is None and agent.api_mode in _CONTINUABLE_MODES
    ):
        escalated = _escalate_reasoning_only(st, _retry, _trunc_msg, _trunc_content)
        if escalated is not None:
            return escalated

    abort = _abort_reason(agent, _trunc_content, _trunc_has_tool_calls)
    if abort is not None:
        line, user_response, error = abort
        agent._vprint(f"{agent.log_prefix}{line}", force=True, diagnostic=True)
        return st.end_turn(user_response, error)

    if agent.api_mode in _CONTINUABLE_MODES:
        cf = _content_filter_fallback(st, _retry)
        if cf is not None:
            return cf
        if _trunc_msg is not None:
            if not _trunc_has_tool_calls:
                return _continue_text(st, _retry, _trunc_msg)
            return _retry_truncated_tool_call(st, _retry, api_kwargs)

    if len(messages) > 1:
        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn", diagnostic=True)
        return st.end_turn(
            _TRUNCATED_FINAL, result_messages=agent._get_messages_up_to_last_assistant(messages)
        )
    # First message was truncated - mark as failed
    agent._flush_status_buffer()
    agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True, diagnostic=True)
    return st.end_turn(_FIRST_TRUNCATED_FINAL, cleanup=False, failed=True)


_CODEX_REPLAY_KEYS = (
    "content", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
)

# Third return value of ``continue_codex_incomplete``: the reasoning-only stall was handed to a
# fallback provider — the caller re-syncs the system prompt identity and continues the turn.
CODEX_FALLBACK_ACTIVATED = "codex_fallback_activated"


def continue_codex_incomplete(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: list[dict[str, Any]],
    conversation_history: Any, api_call_count: int, response: Any = None,
) -> Optional[Any]:
    """Codex Responses ``status=incomplete`` continuation (max 3 per turn).

    Appends the interim assistant message (deduped on visible content only — opaque
    provider state drifts per continuation; ``codex_reasoning_items`` are merged, not
    overwritten, because the earlier response holds the only native-compaction
    checkpoint) and, when a bare retry would be byte-identical, a user-role nudge — only
    after an assistant row, to preserve role alternation. Returns ``None`` to continue
    the turn loop, ``CODEX_FALLBACK_ACTIVATED`` when a reasoning-only stall was handed to
    the next fallback provider, or the terminal ``partial`` result once retries are exhausted.

    Reasoning-only stall ladder (#67321): a response with neither visible text nor a tool
    call advances ``_codex_reasoning_only_streak`` (a visible partial resets it; the aggregate
    ``_codex_incomplete_retries`` stays the cap for partials). Encrypted reasoning replays
    byte-for-byte, so after replay (1) and nudge (2) the third consecutive reasoning-only
    response goes to the configured fallback with the semantic ``incomplete_response`` reason
    instead of ending on the sentinel; when that response consumed the last iteration the
    fallback gets exactly one grace call (``_budget_grace_call`` is consumed by the next
    iteration, and the streak restarts from 0, so a second grace call is unreachable).

    When ``response`` hit ``max_output_tokens`` with no visible text (reasoning ate the
    whole budget), the next attempt goes out with reasoning off and a doubled output
    cap — the same one-shot overrides the chat-completions length path uses — because
    re-sending the identical budget and effort re-burns the budget identically (#90393)."""
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE
    from agent.turn_response_check import _codex_finish_reason

    agent._codex_incomplete_retries += 1
    n = agent._codex_incomplete_retries

    interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
    interim_has_content = bool((interim_msg.get("content") or "").strip())
    _reasoning = interim_msg.get("reasoning")
    interim_has_reasoning = isinstance(_reasoning, str) and bool(_reasoning.strip())
    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
    interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))
    reasoning_only = not interim_has_content and not getattr(assistant_message, "tool_calls", None)
    agent._codex_reasoning_only_streak = agent._codex_reasoning_only_streak + 1 if reasoning_only else 0
    streak = agent._codex_reasoning_only_streak

    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning or interim_has_codex_message_items:
        last_msg = messages[-1] if messages else None
        last_is_dict = isinstance(last_msg, dict)
        last_interim_visible = agent._interim_assistant_visible_text(last_msg) if last_is_dict else ""
        current_interim_visible = agent._interim_assistant_visible_text(interim_msg)
        if last_interim_visible or current_interim_visible:
            same_visible_output = last_interim_visible == current_interim_visible
        else:
            # Neither has text eligible for interim delivery: compare raw content+reasoning.
            same_visible_output = last_is_dict and (
                (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
            )
        if (
            last_is_dict
            and last_msg.get("role") == "assistant"
            and last_msg.get("finish_reason") == "incomplete"
            and same_visible_output
        ):
            # Duplicate: refresh replay state in place, no re-emitted commentary.
            for _key in _CODEX_REPLAY_KEYS:
                if _key not in interim_msg:
                    continue
                if _key == "codex_reasoning_items":
                    from agent.native_compaction import merge_interim_reasoning_items
                    last_msg[_key] = merge_interim_reasoning_items(last_msg.get(_key), interim_msg[_key])
                    # The refreshed items are this response's: so is their replay price.
                    from agent.message_metadata import stamp_reasoning_tokens, stamped_reasoning_tokens
                    stamp_reasoning_tokens(last_msg, stamped_reasoning_tokens(interim_msg))
                else:
                    last_msg[_key] = interim_msg[_key]
        else:
            append_message(messages, interim_msg)
            agent._emit_interim_assistant_message(interim_msg)

    if reasoning_only and streak >= 3:
        if agent._try_activate_fallback(reason=FailoverReason.incomplete_response):
            logger.warning("%sCodex reasoning-only stall after %d attempts — switched to fallback", agent.log_prefix, streak)
            # The trigger may have consumed the turn budget; without a grace call the loop
            # exits before the fallback is ever asked.
            if api_call_count >= agent.max_iterations or agent.iteration_budget.remaining <= 0:
                agent._budget_grace_call = True
            agent._codex_incomplete_retries = 0
            agent._codex_reasoning_only_streak = 0
            if not agent.quiet_mode:
                agent._vprint(
                    f"{agent.log_prefix}↻ Codex reasoning-only stall after {streak} attempts — "
                    f"switching to fallback {agent.model} ({agent.provider})", diagnostic=True,
                )
            agent._emit_diagnostic_wait("↻ model stuck on internal reasoning — switching to fallback provider")
            agent._session_messages = messages
            return CODEX_FALLBACK_ACTIVATED
        # No fallback left: fall through to the terminal sentinel.
    elif n < 3 or reasoning_only:
        # A reasoning-only streak below 3 continues even once partials used up the aggregate
        # cap, so the mixed partial-then-stall variant reaches the ladder above.
        # If the interim has nothing the Responses converter will replay, a bare retry is
        # byte-identical; a replayable interim holding only a ``compaction`` checkpoint
        # ALSO re-sends identically. One bare retry, then always nudge.
        interim_replayable = interim_has_content or interim_has_codex_reasoning or interim_has_codex_message_items
        if not interim_replayable or n >= 2:
            _last_msg = messages[-1] if messages else None
            if isinstance(_last_msg, dict):
                _already_nudged = (
                    _last_msg.get("role") == "user" and _last_msg.get("content") == _CODEX_INCOMPLETE_NUDGE
                )
                # Alternation guard: the nudge may only follow an assistant row.
                if not _already_nudged and _last_msg.get("role") == "assistant":
                    append_message(messages, {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE})
        if not interim_has_content and _codex_finish_reason(response) == "incomplete":
            agent._ephemeral_reasoning_off = True
            # No configured cap means the provider's own ceiling was hit: the observed
            # output_tokens IS that ceiling, so seed the escalation from it (else 4096).
            usage = getattr(response, "usage", None)
            observed = getattr(usage, "output_tokens", None) if not isinstance(usage, dict) else usage.get("output_tokens")
            agent._ephemeral_max_output_tokens = boosted_output_cap(
                agent, None, n, base=agent.max_tokens or int(observed or 0) or 4096
            )
        if not agent.quiet_mode:
            agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({n}/3)", diagnostic=True)
        # Spinner/heartbeat notice: these retries can take minutes and otherwise look
        # like infinite thinking.
        # #70773: same FD-recycle corruption vector as #67142. The shared OpenAI client's connection pool
        # must NOT be closed from this watchdog/poll thread — worker threads from previous stale-killed
        # attempts may still be unwinding their SSL BIOs. The request-local client is already closed above
        # via _close_request_client_once. The shared client will be replaced lazily by
        # _ensure_primary_openai_client on the next request.
        # Surface the continuation on the live spinner/status line (CLI/TUI/Desktop) and gateway heartbeat:
        # each of these retries can spend minutes waiting on the provider, and without a distinct notice the
        # user only sees a generic thinking spinner ("infinite thinking", #64434).
        agent._emit_diagnostic_wait(
            f"↻ model returned reasoning with no final answer — asking it to continue ({n}/3)"
        )
        agent._session_messages = messages
        return None

    agent._codex_incomplete_retries = 0
    agent._codex_reasoning_only_streak = 0
    agent._persist_session(messages, conversation_history)
    return partial_result(
        messages, api_call_count, "Codex response remained incomplete after 3 continuation attempts"
    )


@dataclass
class RefusalVerdict:
    """Outcome of ``handle_content_policy_refusal``: ``"break"`` (fallback activated —
    restart armed on ``_retry``; caller resets retry/compression counters) or
    ``"return"`` (the typed content-policy result in ``result``). ``active_system_prompt``
    is the possibly re-synced system prompt."""

    action: str
    result: Optional[dict[str, Any]]
    active_system_prompt: Any


def handle_content_policy_refusal(
    agent: Any, response: Any, _retry: TurnRetryState, *, thinking_spinner: Any,
    messages: list[dict[str, Any]], api_messages: Any, api_kwargs: Any, active_system_prompt: Any,
    conversation_history: Any, api_call_count: int, effective_task_id: Any, turn_id: Any,
    api_request_id: Any, api_start_time: float, retry_count: int, max_retries: int,
    current_turn_user_idx: Any = None,
) -> RefusalVerdict:
    """HTTP-200 refusal (``finish_reason`` ``content_filter`` / ``guardrail_intervened``).
    Deterministic for the unchanged prompt — never retried: one configured-fallback try,
    else surface the refusal (explanation may live only in the reasoning channel)."""
    from agent.conversation_loop import _arm_fallback_restart, _content_policy_blocked_result

    _refusal_result = normalize_response_for_agent(agent, response)
    _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
    if not _refusal_text:
        _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()
    # Anthropic stop_reason=refusal carries its reason on stop_details (category + optional explanation),
    # not in a content block — without it a classifier halt reads as "(no text)" (#113689).
    _stop_details = (getattr(_refusal_result, "provider_data", None) or {}).get("stop_details")
    if not _refusal_text and isinstance(_stop_details, dict):
        _refusal_text = str(_stop_details.get("explanation") or "").strip() or (
            f"provider refusal category: {_stop_details['category']}" if _stop_details.get("category") else ""
        )

    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type="ContentPolicyBlocked",
        error_message=_refusal_text or "model declined to respond (content_filter)",
        status_code=None, retry_count=retry_count, max_retries=max_retries, retryable=False,
        reason=FailoverReason.content_policy_blocked.value,
    )
    stop_thinking_spinner(agent, thinking_spinner)

    _refusal_log = _refusal_text[:500] + "..." if len(_refusal_text) > 500 else _refusal_text
    # native_stop_reason tells an Anthropic classifier refusal (``refusal``) from a Bedrock guardrail
    # block (``end_turn``); both arrive here as content_filter. Logged before the fallback try so a
    # refusal the fallback answers still leaves its cause in agent.log (#124874).
    logger.warning(
        "%sModel declined to respond (finish_reason=content_filter). model=%s provider=%s "
        "native_stop_reason=%s stop_details=%s refusal=%s",
        agent.log_prefix, agent.model, agent.provider,
        getattr(response, "stop_reason", None) or "n/a", str(_stop_details or "n/a")[:500],
        _refusal_log or "(no text)",
    )
    if agent._has_pending_fallback():
        agent._buffer_diagnostic_status("⚠️ Model declined to respond (safety refusal) — trying fallback...")
    # Name the cause: without a reason the switch notice reads "unavailable (provider failure)",
    # while the same refusal raised as an HTTP 400 (DeepSeek's "Content Exists Risk") says
    # "content policy blocked the request" (#124874). Not a rate-limit reason, so no cooldown is armed.
    if agent._try_activate_fallback(reason=FailoverReason.content_policy_blocked):
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        return RefusalVerdict("break", None, active_system_prompt)

    agent._flush_status_buffer()
    agent._emit_diagnostic_status("⚠️ The model declined to respond to this request (safety refusal).")
    _refusal_response = "⚠️ " + content_policy_copy(
        label=provider_label_for(agent.provider),
        summary=_refusal_text or "the model returned no explanation",
    )
    agent._cleanup_task_resources(effective_task_id)
    # Do not persist the refused turn. A safety refusal is deterministic for the unchanged
    # conversation (the filter judges the whole thread), so a durable refused turn re-trips the
    # filter on every later message and locks the chat — the "stuck session" report. Hand the caller
    # the PRE-refusal history so an in-memory surface (CLI/TUI/ACP keep ``result["messages"]`` as the
    # live history) carries nothing forward, and flush token usage for the call that was made. The
    # gateway reloads history from the durable transcript each turn, so it also drops the refused
    # turn owner-side after the lease releases (``_drop_content_policy_refused_turn`` in
    # run_turn_persistence) — a transcript mutation is refused while the turn lease is held, and a
    # managed worker never mutates its own transcript. Mirrors the context_overflow no-grow rule
    # (#1630); transient failures keep their persist-and-replay path, only a non-retryable block is
    # dropped.
    _db = getattr(agent, "_session_db", None)
    if _db is not None:
        with suppress(Exception):
            _db.flush_token_counts()
    # Hand back the CLEAN pre-refusal history, not the loop's ``conversation_history``: an in-place
    # compaction this turn rewrote that baseline into ``list(messages)`` — the whole poisoned working
    # set including the refused request + tool rounds — so returning it would re-poison CLI/ACP memory
    # even after the owner cleans the durable transcript (#149708 review). Build it from the real
    # turn boundary, and carry the refused run's durable boundary row so the owner drops the exact turn.
    from agent.turn_context import history_before_current_turn, refused_turn_boundary_row_id
    clean_history = history_before_current_turn(messages, current_turn_user_idx, conversation_history)
    _boundary_row_id = refused_turn_boundary_row_id(messages, current_turn_user_idx)
    return RefusalVerdict("return", _content_policy_blocked_result(
        clean_history, api_call_count, final_response=_refusal_response,
        error_detail=_refusal_text or "model declined (content_filter)",
        boundary_row_id=_boundary_row_id,
    ), active_system_prompt)
