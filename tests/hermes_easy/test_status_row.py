from types import SimpleNamespace
from unittest.mock import patch

from hermes_easy import layout


def render(width=80, bypass=False, mode="manual", model="Qwen", rate=None, rate_model="Qwen"):
    cli = SimpleNamespace(
        session_id="test", model=model,
        agent=SimpleNamespace(_easy_decode_rate=rate, _easy_decode_model=rate_model),
        _get_tui_terminal_width=lambda: width,
        _status_bar_display_width=len,
        _trim_status_bar_text=lambda text, limit: text[:limit],
    )
    ui_values = {"status.line": "standard", "status.key_hints": False}
    with patch.object(layout.ui, "get", side_effect=ui_values.__getitem__), \
         patch("tools.approval.is_approval_bypass_active_for_session", return_value=bypass), \
         patch("tools.approval_context._get_approval_mode", return_value=mode):
        return "".join(text for _, text in layout.status_line_fragments(cli)).strip()


def test_status_reports_effective_bypass_and_omits_unavailable_rate():
    assert render(bypass=True) == "Approval bypass · Qwen"
    assert render(mode="smart") == "Smart approvals · Qwen"


def test_status_shows_only_the_current_models_decode_rate():
    assert render(rate=17.85) == "Manual approvals · Qwen · 17.9 tok/s"
    assert render(rate=17.85, rate_model="other") == "Manual approvals · Qwen"


def test_status_handles_missing_model_and_narrow_terminal():
    assert "No model selected" in render(model="")
    assert len(render(width=15)) <= 15


def test_friendly_local_model_name():
    assert layout.display_model("qwen3.8-27b-agent") == "Qwen 3.8 27B agent"
