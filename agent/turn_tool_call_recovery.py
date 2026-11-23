"""Recovery for a tool call whose arguments were cut off (output cap, full context slot, dropped stream).

A truncated call is NEVER run and never stored in history, so nothing can execute twice and every
earlier tool result stays in the session. What this module decides is how the *next request* differs
from the one that failed, because re-sending it unchanged re-truncates in exactly the same place:

1. Classify the stop: the context slot filled (prompt + completion reached the window), the output
   cap was hit, or the stream dropped. Only the first two can be fixed by changing the request.
2. Learn the real window. llama.cpp routers report the *trained* window in ``/models`` (262,144 for a
   Qwen3-Coder) while a slot holds a quarter of it; when the stop shows the slot was full, the
   compressor's window is corrected (server ``/props`` first, else what the stop itself proves) so the
   next iteration's preflight compresses instead of re-sending a prompt that leaves no room to write.
3. Ask for a smaller, complete call: a request-only nudge (never stored) tells the model its call was
   not run and to split large writes, and the output cap is raised only as far as the window allows.
4. Stop only when the ladder is spent, with copy that names the real cause and a concrete next step.

Both detection sites (``turn_truncation`` for ``finish_reason == "length"`` and
``turn_tool_validation`` for routers that rewrite it to ``tool_calls``) call ``plan_recovery``.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.usage_pricing import normalize_usage

logger = logging.getLogger("agent.conversation_loop")

MAX_ATTEMPTS = 6
# A slot whose prompt + completion reached this share of the window stopped because it was full.
_FULL_SLOT_SHARE = 0.985
# Output room worth retrying into; below it the prompt itself must shrink first.
_MIN_USEFUL_OUTPUT = 3072
_CAP_HEADROOM = 256
# Largest output cap the ladder offers when the model advertises no limit of its own.
_CAP_CEILING = 131072
# Floor for a learned window: llama.cpp slots are never this small for a 64K-minimum agent, so a
# smaller observation is a stream cut, not a window.
_MIN_LEARNED_WINDOW = 16384

CONTEXT_FULL = "context_full"
OUTPUT_CAP = "output_cap"
STREAM_DROP = "stream_drop"
UNKNOWN = "unknown"
UNREPORTED = "unreported"  # args cut mid-JSON but the server never reported hitting a limit (router/transport)

_SMALLER_PIECES_NUDGE = (
    "[System: Your previous tool call was cut off before its arguments were complete, so it was NOT run "
    "and nothing changed. Do not send it again unchanged. Make the action smaller: write large files in "
    "several calls (create the file with the first part, then add the rest with separate calls), prefer a "
    "targeted patch over rewriting a whole file, and keep each call's arguments under about {lines} lines.]"
)


@dataclass
class LengthStop:
    """Why one response ended early, measured from its own usage and the request that produced it."""

    kind: str = UNKNOWN
    prompt_tokens: int = 0
    completion_tokens: int = 0
    window: int = 0
    requested_cap: int = 0

    @property
    def headroom(self) -> int:
        """Tokens of the window left for output after the prompt (0 when the window is unknown)."""
        return max(self.window - self.prompt_tokens, 0) if self.window else 0


@dataclass
class RecoveryPlan:
    """``retry``: send a changed request; otherwise end the turn with ``message``."""

    retry: bool
    stop: LengthStop
    steps: list[str] = field(default_factory=list)
    message: str = ""
    error: str = ""
    cap: Optional[int] = None  # output cap armed for the retry (also in ``agent._ephemeral_max_output_tokens``)


def _usage_tokens(agent: Any, response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage:
        with suppress(Exception):
            canonical = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode)
            return int(canonical.prompt_tokens or 0), int(canonical.output_tokens or 0)
    compressor = getattr(agent, "context_compressor", None)
    return (int(getattr(compressor, "last_prompt_tokens", 0) or 0),
            int(getattr(compressor, "last_completion_tokens", 0) or 0))


def _window_of(agent: Any) -> int:
    return int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0)


def classify_length_stop(agent: Any, response: Any = None, api_kwargs: Any = None, *, stream_drop: bool = False,
                         reported: bool = True) -> LengthStop:
    """Measure a truncated response. ``response`` may be None (validation path): the compressor's last
    usage stands in for it. ``reported`` is False when the server never said ``length`` (a router rewrote
    it): such a stop is blamed on neither the output cap nor the window unless the numbers prove it."""
    prompt, completion = _usage_tokens(agent, response)
    window = _window_of(agent)
    requested = 0
    with suppress(Exception):
        requested = int(agent._requested_output_cap_from_api_kwargs(api_kwargs) or 0)
    stop = LengthStop(prompt_tokens=prompt, completion_tokens=completion, window=window, requested_cap=requested)
    if stream_drop:
        stop.kind = STREAM_DROP
    elif window and prompt and prompt + completion >= window * _FULL_SLOT_SHARE:
        stop.kind = CONTEXT_FULL
    elif prompt and window and window - prompt < _MIN_USEFUL_OUTPUT:
        stop.kind = CONTEXT_FULL  # nothing useful fits even if the server had not cut it
    elif requested and completion >= requested * 0.98:
        stop.kind = OUTPUT_CAP
    elif not reported:
        stop.kind = UNREPORTED
    elif response is None or not completion:
        stop.kind = UNKNOWN
    else:
        stop.kind = OUTPUT_CAP
    return stop


def _server_window(agent: Any) -> Optional[int]:
    """The window the server allocates right now (llama.cpp ``/props`` per slot), bypassing the memo."""
    from agent.model_metadata import fetch_endpoint_model_metadata

    base_url = getattr(agent, "base_url", "") or ""
    model = getattr(agent, "model", "") or ""
    if not (base_url and model):
        return None
    with suppress(Exception):
        meta = fetch_endpoint_model_metadata(base_url, api_key=getattr(agent, "api_key", "") or "", force_refresh=True)
        ctx = (meta.get(model) or {}).get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            return ctx
    return None


def _window_proven_by(stop: LengthStop) -> Optional[int]:
    """The window a stop proves when no server will say. A slot that fills ends exactly at its
    allocation (llama.cpp ``n_ctx`` is a multiple of 256), so prompt + completion on a stop the output
    cap did not cause is the window. Anything else could be a stream cut, which proves nothing."""
    total = stop.prompt_tokens + stop.completion_tokens
    cap_caused_it = bool(stop.requested_cap) and stop.completion_tokens >= stop.requested_cap * 0.98
    if stop.kind == STREAM_DROP or cap_caused_it or total < _MIN_LEARNED_WINDOW or total % 256:
        return None
    return total


def adopt_real_window(agent: Any, stop: LengthStop) -> Optional[int]:
    """Correct the compressor's window when it is larger than the slot really is; returns the new
    window, or None when it was already right. Server answer first, then what the stop proves."""
    compressor = getattr(agent, "context_compressor", None)
    assumed = _window_of(agent)
    if compressor is None or not assumed:
        return None
    real = _server_window(agent)
    if real is None:
        real = _window_proven_by(stop)
    if not real or real >= assumed:
        return None
    compressor.update_model(
        agent.model, real, base_url=getattr(agent, "base_url", "") or "",
        api_key=getattr(agent, "api_key", "") or "", provider=getattr(agent, "provider", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "",
    )
    with suppress(Exception):
        from agent.model_metadata import save_context_length
        save_context_length(agent.model, getattr(agent, "base_url", "") or "", real)
    logger.warning("Context window corrected %s -> %s tokens for %s after a full-slot truncation",
                   f"{assumed:,}", f"{real:,}", agent.model)
    stop.window = real
    return real


def _cap_for(agent: Any, stop: LengthStop, attempt: int) -> Optional[int]:
    """Output cap for this attempt: the shared truncation ladder (``boosted_output_cap``: doubles per
    attempt, never above the model's known limit), then held inside the window. None when no honest
    larger cap exists (a made-up one could LOWER a server's own default)."""
    from agent.turn_truncation import boosted_output_cap

    configured = agent.max_tokens if isinstance(getattr(agent, "max_tokens", None), int) else 0
    base = configured or stop.requested_cap or stop.completion_tokens or 4096
    cap = boosted_output_cap(agent, stop.requested_cap or None, attempt, base=base)
    cap = min(cap, max(_CAP_CEILING, stop.requested_cap))  # doubling stops at a sane ceiling, never below the request
    if stop.window and stop.prompt_tokens:
        cap = min(cap, stop.window - stop.prompt_tokens - _CAP_HEADROOM)
    # At the model's own ceiling the ladder returns the cap it was given: re-sending it (with the smaller-pieces
    # nudge) is still the right move, so equality with the request is allowed, never a drop below it.
    return cap if cap >= stop.requested_cap and cap > max(stop.completion_tokens, 1024) else None


def _reset(agent: Any) -> None:
    agent._tool_trunc_attempts = 0
    agent._tool_trunc_tried = []


def reset_tool_call_recovery(agent: Any) -> None:
    """Forget the ladder: a turn started, or a tool batch ran (progress was made)."""
    _reset(agent)


def plan_recovery(agent: Any, response: Any = None, api_kwargs: Any = None, *, stream_drop: bool = False,
                  reported: bool = True) -> RecoveryPlan:
    """Decide what the next request should look like after a truncated tool call, and arm it.

    Arming sets the one-shot output cap and request-only nudge the request builder consumes; the caller
    restarts the iteration so preflight compression sees the corrected window."""
    attempt = int(getattr(agent, "_tool_trunc_attempts", 0) or 0) + 1
    agent._tool_trunc_attempts = attempt
    tried: list[str] = list(getattr(agent, "_tool_trunc_tried", None) or [])
    stop = classify_length_stop(agent, response, api_kwargs, stream_drop=stream_drop, reported=reported)

    corrected = adopt_real_window(agent, stop) if stop.kind != STREAM_DROP else None
    if corrected:
        stop.kind = CONTEXT_FULL if stop.prompt_tokens + stop.completion_tokens >= corrected * 0.9 else stop.kind
    steps: list[str] = []
    if corrected:
        steps.append(f"context window corrected to {corrected:,} tokens")

    if attempt > MAX_ATTEMPTS:
        return _give_up(agent, stop, tried, attempt - 1)

    if stop.kind == CONTEXT_FULL:
        steps.append("compacting the conversation to make room")
        # Under the corrected window the prompt is over the compression threshold, so the restarted
        # iteration's preflight pass compresses. Without a server window the stop itself shrank it.
    # Always ask for a smaller call: even when compaction frees the room, a call this large is the one
    # most likely to be cut again, and the nudge costs a few tokens at the tail of one request.
    if stop.kind != STREAM_DROP:
        lines = max(40, 160 // (2 ** (attempt - 1)))
        agent._request_only_nudge = _SMALLER_PIECES_NUDGE.format(lines=lines)
        steps.append(f"asked for smaller pieces (about {lines} lines per call)")
    cap = _cap_for(agent, stop, attempt) if stop.kind != STREAM_DROP else None
    if cap:
        agent._ephemeral_max_output_tokens = cap
        steps.append(f"output limit {cap:,} tokens")
    if attempt >= 3:
        agent._ephemeral_reasoning_off = True
        steps.append("reasoning off for this call")

    tried.append("; ".join(steps) or "plain retry")
    agent._tool_trunc_tried = tried
    return RecoveryPlan(retry=True, stop=stop, steps=steps, cap=cap)


def _give_up(agent: Any, stop: LengthStop, tried: list[str], attempts: int) -> RecoveryPlan:
    window = f"{stop.window:,}" if stop.window else "its"
    ladder = "\n".join(f"{i}. {t}" for i, t in enumerate(tried, start=1))
    if stop.kind == CONTEXT_FULL:
        cause = (
            f"The conversation fills this model's {window}-token context window, so there is no room left "
            "to write the next action. This is a context limit, not an output limit."
        )
        advice = "Run `/compress`, or `/new` to start fresh (your files and the work done so far are untouched)."
    elif stop.kind == STREAM_DROP:
        cause = "The connection to the model kept dropping while it was writing the action."
        advice = "Check the model server, then send `/retry`."
    elif stop.kind == UNREPORTED:
        cause = (
            f"The action arrived cut off on every one of {attempts} attempts, but the server never reported hitting "
            "a limit, so this is most likely a dropped connection or a provider/router cutting the reply short."
        )
        advice = "Check the connection or the router in front of the model, then send `/retry`."
    else:
        cause = (
            f"The model hit its output length limit on every one of {attempts} attempts"
            + (f" (about {stop.completion_tokens:,} tokens each)" if stop.completion_tokens else "")
            + ", even asked for smaller pieces."
        )
        advice = "Raise the output limit on the model server, or switch model with `/model`, then send `/retry`."
    message = (
        f"⚠️ **The action could not be completed.** {cause}\n\nNothing was run, and nothing the agent had already "
        f"done was lost. What was tried:\n{ladder}\n\n{advice}"
    )
    _reset(agent)
    agent._ephemeral_reasoning_off = False
    return RecoveryPlan(retry=False, stop=stop, steps=tried, message=message,
                        error=f"Tool call truncated on {attempts} consecutive attempts ({stop.kind})")
