"""llama.cpp context-window probes (sibling of ``model_metadata``)."""

from __future__ import annotations

from itertools import pairwise
from typing import Optional
from urllib.parse import quote


def _json_object(resp) -> dict:
    """The response body when it is a JSON object, else an empty dict."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _router_child_ctx(client, base: str, child: dict) -> Optional[int]:
    """The per-request window of one router child: ``/props`` n_ctx when it is loaded, else what its
    launch args allocate (``--ctx-size`` split across ``--parallel`` slots). Never autoloads."""
    from agent.model_metadata import _positive_int

    child_id = child.get("id")
    status = child.get("status") if isinstance(child.get("status"), dict) else {}
    if status.get("value") in ("loaded", "ready"):
        try:
            resp = client.get(f"{base}/props?model={quote(str(child_id))}&autoload=false")
        except Exception:
            resp = None
        if resp is not None and resp.status_code == 200:
            settings = _json_object(resp).get("default_generation_settings")
            n_ctx = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
            if n_ctx:
                return n_ctx
    return _launch_args_ctx(status.get("args"))


def _launch_args_ctx(raw_args) -> Optional[int]:
    """``--ctx-size`` divided by ``--parallel`` from a router child's args (llama.cpp gives each slot an
    equal share unless ``--kv-unified`` pools them), or None when the args carry no ctx-size."""
    args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
    flags = dict(pairwise(args))
    ctx = next((int(flags[f]) for f in ("--ctx-size", "-c") if flags.get(f, "").isdigit() and int(flags[f]) > 0), None)
    if ctx is None:
        return None
    slots = next((int(flags[f]) for f in ("--parallel", "-np") if flags.get(f, "").isdigit() and int(flags[f]) > 0), 1)
    return ctx if "--kv-unified" in args or slots <= 1 else ctx // slots


def apply_llamacpp_props(cache: dict, request_candidate: str, headers: dict, verify) -> None:
    """Overwrite ``context_length`` in the /models cache with the window a request really gets.

    ``/models`` reports the TRAINED window (262,144 for a Qwen3-Coder), which is 2-4x what a router
    slot allocates; trusting it starves the compressor and lets the slot fill mid tool call (#runtime
    recovery). A single server answers ``/props`` with its own n_ctx. A router answers ``/props`` with
    ``role: router`` and ``n_ctx: 0`` (older builds 400ed), so every child is read separately."""
    from agent.model_metadata import model_metadata_http

    base = request_candidate.rstrip("/").replace("/v1", "")

    def _get(path: str):
        resp = model_metadata_http.get(base + path, headers=headers, timeout=5, verify=verify)
        return resp

    props_resp = _get("/v1/props")  # newer single servers; routers and older builds answer /props only
    if props_resp.is_error:
        props_resp = _get("/props")
    if not props_resp.is_error:
        props = props_resp.json()
        if isinstance(props, dict) and props.get("role") != "router":
            settings = props.get("default_generation_settings") or {}
            n_ctx, alias = settings.get("n_ctx"), props.get("model_alias", "")
            if n_ctx and alias and alias in cache:
                cache[alias]["context_length"] = n_ctx
            return
    native = _get("/models")
    if native.is_error:
        return

    class _Client:  # adapts the shared metadata HTTP helper to the .get(url) shape the probes use
        @staticmethod
        def get(url):
            return model_metadata_http.get(url, headers=headers, timeout=5, verify=verify)

    for child in (native.json() or {}).get("data", [])[:64]:
        if not isinstance(child, dict) or child.get("id") not in cache:
            continue
        child_ctx = _router_child_ctx(_Client, base, child)
        if child_ctx:
            cache[child["id"]]["context_length"] = child_ctx


def llamacpp_context(client, server_url: str, model: str) -> Optional[int]:
    """llama.cpp's RUNTIME n_ctx: /props for a loaded model, else the ``--ctx-size`` the router will
    launch it with (its /models entry), so a lazily-loaded model doesn't fall to a family catch-all.
    The /props read passes autoload=false: the router loads any model a request names, and a context
    probe must not undo an Eject or an idle unload."""
    import httpx

    from agent.model_metadata import _positive_int

    for props_path in (f"/props?model={quote(model)}&autoload=false", "/props"):
        try:
            resp = client.get(f"{server_url}{props_path}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            continue
        settings = _json_object(resp).get("default_generation_settings")
        n_ctx = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
        if n_ctx is not None:
            return n_ctx
    return router_launch_ctx(client, server_url, model)


def router_launch_ctx(client, server_url: str, model: str) -> Optional[int]:
    """``--ctx-size`` from the router's launch args for *model* (GET /models), or None."""
    import httpx

    from agent.model_metadata import _model_id_matches

    try:
        resp = client.get(f"{server_url}/models")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    entries = _json_object(resp).get("data")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not _model_id_matches(str(entry.get("id", "")), model):
            continue
        status = entry.get("status")
        return _launch_args_ctx(status.get("args") if isinstance(status, dict) else None)
    return None
