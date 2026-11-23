"""Non-blocking, read-only status for local llama-server routers."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)


def local_router_url(base_url):
    try:
        parts = urlsplit(base_url or "")
        local = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_private
    except ValueError:
        local = False
    if parts.scheme not in ("http", "https") or not local or not parts.hostname:
        return None
    path = parts.path.rstrip("/")
    path = path.removesuffix("/v1")
    return urlunsplit((parts.scheme, parts.netloc, path + "/models", "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_local(request, timeout):
    # Never forward provider credentials across a redirect or a configured proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    return opener.open(request, timeout=timeout)


def probe(base_url, model, api_key=None, opener=_open_local):
    if not model:
        return "unselected"
    url = local_router_url(base_url)
    if not url:
        return "unknown"
    headers = {"Authorization": "Bearer " + api_key} if isinstance(api_key, str) and api_key else {}
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=3) as response:
            data = json.load(response)
        items = data.get("data", [])
        for item in items:
            if item.get("id") == model or model in (item.get("aliases") or []):
                status = item.get("status")
                value = status.get("value") if isinstance(status, dict) else None
                return {"loaded": "ready", "loading": "loading", "unloaded": "unloaded",
                        "unloading": "unloaded", "failed": "failed", "error": "failed"}.get(value, "unknown")
        # Other OpenAI-compatible APIs may publish a partial catalog without load states.
        return "missing" if any(isinstance(item.get("status"), dict) for item in items) else "unknown"
    except urllib.error.HTTPError:
        log.debug("Model status request rejected", exc_info=True)
        return "unknown"
    except (OSError, ValueError, TypeError, AttributeError):
        log.debug("Model status request failed", exc_info=True)
        return "disconnected"


def run(cli):
    """Poll without loading models; only invalidate on a state change."""
    while not getattr(cli, "_should_exit", False):
        model = getattr(cli, "model", "") or ""
        base = getattr(cli, "base_url", None)
        key = getattr(cli, "api_key", None)
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            provider = (cfg.get("providers") or {}).get(getattr(cli, "provider", ""), {})
            if isinstance(provider, dict):
                key = os.getenv(provider.get("key_env", "")) or key
        except Exception:
            log.debug("Could not resolve model status credentials", exc_info=True)
        state = probe(base, model, key)
        old = getattr(cli, "_easy_model_status", None)
        # A previous model's status must not be shown after a switch.
        current = (model, base, state)
        if current != old:
            cli._easy_model_status = current
            if getattr(cli, "_app", None):
                cli._invalidate(min_interval=0.0)
        time.sleep(3)


def decode_rate(chunk):
    """Read llama-server's measured decode tokens and milliseconds from an SSE chunk."""
    extra = getattr(chunk, "model_extra", None) or {}
    timings = extra.get("timings") or getattr(chunk, "timings", None)
    if not isinstance(timings, dict):
        return None
    count, ms = timings.get("predicted_n"), timings.get("predicted_ms")
    if (isinstance(count, (int, float)) and not isinstance(count, bool)
            and isinstance(ms, (int, float)) and not isinstance(ms, bool)
            and 0 < count < 10**9 and 0 < ms < 10**12):
        return count * 1000.0 / ms
    return None
