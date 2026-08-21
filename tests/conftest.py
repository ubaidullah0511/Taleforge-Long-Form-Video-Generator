import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _disable_ai_generation_fallback_by_default():
    """The test suite must stay hermetic regardless of whatever
    settings.enable_ai_generation_fallback defaults to for real runs — a
    test that doesn't specifically exercise AI generation must never trigger
    a real network call to OpenAI's image API just because the app-wide
    default happens to be True. Tests that DO want the AI-generation path
    re-enable it themselves via their own patch.object(settings,
    "enable_ai_generation_fallback", True)."""
    original = settings.enable_ai_generation_fallback
    settings.enable_ai_generation_fallback = False
    yield
    settings.enable_ai_generation_fallback = original
