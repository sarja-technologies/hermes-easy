import io
import json
from urllib.error import URLError

from hermes_easy.model_status import probe, local_router_url


def opener(data):
    def open_request(request, timeout):
        return io.BytesIO(json.dumps(data).encode())
    return open_request


def test_router_states_and_switching():
    models = {"data": [{"id": "a", "status": {"value": "loading"}},
                       {"id": "b", "status": {"value": "loaded"}}]}
    assert probe("http://127.0.0.1:8080/v1", "a", opener=opener(models)) == "loading"
    assert probe("http://127.0.0.1:8080/v1", "b", opener=opener(models)) == "ready"
    assert probe("http://127.0.0.1:8080/v1", "c", opener=opener(models)) == "missing"
    assert probe("http://127.0.0.1:8080/v1", "", opener=opener(models)) == "unselected"


def test_disconnected_and_remote_url():
    def failed(*args, **kwargs):
        raise URLError("offline")
    assert probe("http://127.0.0.1:8080/v1", "a", opener=failed) == "disconnected"
    assert local_router_url("https://example.com/v1") is None


def test_decode_rate_uses_backend_count_and_decode_duration():
    from types import SimpleNamespace
    from hermes_easy.model_status import decode_rate
    assert decode_rate(SimpleNamespace(model_extra={"timings": {"predicted_n": 40, "predicted_ms": 2000}})) == 20
    assert decode_rate(SimpleNamespace(model_extra={"timings": {"predicted_n": 40, "predicted_ms": 0}})) is None
    assert decode_rate(SimpleNamespace(model_extra={})) is None


def test_loading_failure_and_recovery():
    base = "http://127.0.0.1:8080/v1"
    def state(value):
        return opener({"data": [{"id": "a", "status": {"value": value}}]})
    assert probe(base, "a", opener=state("unloaded")) == "unloaded"
    assert probe(base, "a", opener=state("loading")) == "loading"
    assert probe(base, "a", opener=state("failed")) == "failed"
    assert probe(base, "a", opener=state("loaded")) == "ready"


def test_gateway_usage_preserves_native_decode_rate():
    from types import SimpleNamespace
    from gateway.session_usage import usage_stats, usage_frame
    agent = SimpleNamespace(_easy_decode_rate=20.25, _easy_decode_model="a")
    frame = usage_frame(usage_stats(agent))
    assert frame["decode_tokens_per_s"] == 20.25
    assert frame["decode_model"] == "a"
