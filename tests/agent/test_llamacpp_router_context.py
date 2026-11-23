"""A llama.cpp router lists the TRAINED window in /models and answers a bare /props with
``role: router, n_ctx: 0``. Hermes must read each child's real per-slot window instead, or the
compressor believes in 262,144 tokens while a slot holds 65,536 and long sessions die mid tool call."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.model_metadata_llamacpp import _launch_args_ctx, apply_llamacpp_props

BASE = "http://spark:8080/v1"


def _resp(body, status=200):
    return SimpleNamespace(is_error=status >= 400, status_code=status, json=lambda: body)


def _child(cid, state, ctx, parallel=None, unified=False):
    args = ["llama-server", "--alias", cid, "--ctx-size", str(ctx)]
    if parallel:
        args += ["--parallel", str(parallel)]
    if unified:
        args.append("--kv-unified")
    return {"id": cid, "owned_by": "llamacpp", "status": {"value": state, "args": args}}


def _server(children, props_by_model, bare):
    def get(url, **kwargs):
        path = url.split("8080", 1)[1]
        if path == "/v1/props":
            return _resp({}, 404)
        if path == "/props":
            return _resp(bare)
        if path == "/models":
            return _resp({"data": children})
        if path.startswith("/props?model="):
            model = path.split("model=")[1].split("&")[0]
            return _resp(props_by_model[model]) if model in props_by_model else _resp({}, 404)
        return _resp({}, 404)
    return get


def _cache(*ids):
    return {i: {"context_length": 262144} for i in ids}


class TestRouterWindows:
    def test_loaded_child_uses_props_and_unloaded_uses_launch_args_per_slot(self):
        children = [_child("coder", "loaded", 131072, 2), _child("big", "unloaded", 131072, 2), _child("solo", "unloaded", 131072, 1)]
        props = {"coder": {"default_generation_settings": {"n_ctx": 65536}}}
        bare = {"role": "router", "model_alias": "llama-server", "default_generation_settings": {"n_ctx": 0}}
        cache = _cache("coder", "big", "solo")
        with patch("agent.model_metadata.model_metadata_http.get", side_effect=_server(children, props, bare)):
            apply_llamacpp_props(cache, BASE, {}, True)
        assert {k: v["context_length"] for k, v in cache.items()} == {"coder": 65536, "big": 65536, "solo": 131072}

    def test_older_router_that_errors_on_bare_props_is_handled_the_same(self):
        children = [_child("coder", "loaded", 131072, 2)]
        props = {"coder": {"default_generation_settings": {"n_ctx": 65536}}}
        cache = _cache("coder")

        def get(url, **kw):
            if url.endswith("/props"):  # both /v1/props and /props error: only the per-child read works
                return _resp({}, 400)
            return _server(children, props, {})(url, **kw)

        with patch("agent.model_metadata.model_metadata_http.get", side_effect=get):
            apply_llamacpp_props(cache, BASE, {}, True)
        assert cache["coder"]["context_length"] == 65536

    def test_a_single_server_keeps_its_own_n_ctx(self):
        bare = {"model_alias": "one", "default_generation_settings": {"n_ctx": 32768}}
        cache = _cache("one")
        with patch("agent.model_metadata.model_metadata_http.get", side_effect=_server([], {}, bare)):
            apply_llamacpp_props(cache, BASE, {}, True)
        assert cache["one"]["context_length"] == 32768

    def test_an_unloaded_child_is_never_probed_through_props(self):
        """Probing /props?model= on an unloaded child would load it (memory, minutes): launch args only."""
        seen = []
        base_get = _server([_child("idle", "unloaded", 65536, 1)], {}, {"role": "router", "default_generation_settings": {"n_ctx": 0}})

        def get(url, **kw):
            seen.append(url)
            return base_get(url, **kw)

        cache = _cache("idle")
        with patch("agent.model_metadata.model_metadata_http.get", side_effect=get):
            apply_llamacpp_props(cache, BASE, {}, True)
        assert not any("/props?model=" in u for u in seen)
        assert cache["idle"]["context_length"] == 65536


class TestLaunchArgs:
    def test_ctx_is_split_across_slots(self):
        assert _launch_args_ctx(["--ctx-size", "131072", "--parallel", "2"]) == 65536
        assert _launch_args_ctx(["-c", "65536", "-np", "4"]) == 16384

    def test_unified_kv_pools_the_slots(self):
        assert _launch_args_ctx(["--ctx-size", "131072", "--parallel", "2", "--kv-unified"]) == 131072

    def test_no_ctx_flag_is_unknown(self):
        assert _launch_args_ctx(["--parallel", "2"]) is None
        assert _launch_args_ctx(None) is None
