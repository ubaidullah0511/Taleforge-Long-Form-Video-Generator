"""Offline test for app.style_decision.decide_style — no network/API keys
needed, Groq is mocked. Run with: pytest tests/test_style_decision.py
"""
from unittest.mock import patch

from app.style_decision import StyleDecision, decide_style


def test_decide_style_returns_valid_groq_response():
    data = {"pacing": "fast", "transition_style": "slide", "caption_emphasis": ["war", "victory"]}
    with patch("app.style_decision.generate_json", return_value=data):
        decision = decide_style("some script")
    assert decision == StyleDecision(pacing="fast", transition_style="slide", caption_emphasis=["war", "victory"])


def test_decide_style_normalizes_invalid_pacing_and_transition():
    data = {"pacing": "blazing", "transition_style": "wipe-diagonal", "caption_emphasis": ["x"]}
    with patch("app.style_decision.generate_json", return_value=data):
        decision = decide_style("some script")
    assert decision.pacing == "medium"  # invalid value normalized to default
    assert decision.transition_style == "fade"  # invalid value normalized to default
    assert decision.caption_emphasis == ["x"]


def test_decide_style_falls_back_to_defaults_when_groq_raises():
    with patch("app.style_decision.generate_json", side_effect=RuntimeError("network down")):
        decision = decide_style("some script")
    assert decision == StyleDecision()  # never blocks/raises — sensible hardcoded defaults


def test_decide_style_falls_back_when_response_is_malformed():
    with patch("app.style_decision.generate_json", return_value={"unexpected": "shape"}):
        decision = decide_style("some script")
    assert decision == StyleDecision()


def test_decide_style_caps_and_stringifies_emphasis_list():
    data = {"pacing": "slow", "transition_style": "zoom", "caption_emphasis": list(range(30))}
    with patch("app.style_decision.generate_json", return_value=data):
        decision = decide_style("some script")
    assert len(decision.caption_emphasis) == 15
    assert all(isinstance(w, str) for w in decision.caption_emphasis)


if __name__ == "__main__":
    test_decide_style_returns_valid_groq_response()
    test_decide_style_normalizes_invalid_pacing_and_transition()
    test_decide_style_falls_back_to_defaults_when_groq_raises()
    test_decide_style_falls_back_when_response_is_malformed()
    test_decide_style_caps_and_stringifies_emphasis_list()
    print("OK")
