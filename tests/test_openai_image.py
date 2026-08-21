"""Offline tests for app.stock.openai_image (OpenAI image-generation
fallback) — no real network access (app.llm_client._client is replaced
with a fake). Run with: pytest tests/test_openai_image.py
"""
import asyncio
import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.stock import openai_image
from app.stock.openai_image import generate_fallback_image_openai

_JPEG_BYTES = b"\xff\xd8\xff" + b"x" * 2000
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 2000
_WEBP_BYTES = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"x" * 2000
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/images/generations")


def _status_error(status_code: int, message: str = "error") -> APIStatusError:
    return APIStatusError(message, response=httpx.Response(status_code, request=_REQUEST), body=None)


def _fake_response(image_bytes: bytes):
    response = MagicMock()
    response.data = [MagicMock(b64_json=base64.b64encode(image_bytes).decode())]
    response._request_id = "req_123"
    return response


def _run_generate(images_generate, output_suffix=".jpg", **kwargs):
    fake_client = MagicMock()
    fake_client.images.generate.side_effect = images_generate
    with patch.object(openai_image, "_client", return_value=fake_client):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"generated{output_suffix}"
            success, reason = asyncio.run(
                generate_fallback_image_openai("panic pool owner", output_path, **kwargs)
            )
            content = output_path.read_bytes() if output_path.exists() else None
            leftover_tmp = list(Path(tmpdir).glob("*.tmp"))
    return success, reason, content, fake_client, leftover_tmp


def test_generate_fallback_image_openai_writes_file_and_returns_true_on_success():
    def images_generate(**kw):
        assert kw["model"] == "gpt-image-2"  # default per settings.ai_generation_model
        assert kw["prompt"] == "panic pool owner"
        assert kw["output_format"] == "jpeg"  # inferred from .jpg suffix
        return _fake_response(_JPEG_BYTES)

    success, reason, content, _, leftover_tmp = _run_generate(images_generate)

    assert success is True
    assert reason is None
    assert content == _JPEG_BYTES
    assert leftover_tmp == []  # atomic write leaves no .tmp file behind


def test_generate_fallback_image_openai_passes_configured_quality_and_default_size():
    calls = []

    def images_generate(**kw):
        calls.append(kw)
        return _fake_response(_JPEG_BYTES)

    with patch.object(openai_image.settings, "ai_generation_quality", "low"):
        _run_generate(images_generate)

    assert calls[0]["quality"] == "low"
    assert calls[0]["size"] == "2048x1152"  # gpt-image-2 default -> true 16:9
    assert calls[0]["n"] == 1


def test_generate_fallback_image_openai_uses_legacy_size_for_older_model():
    calls = []

    def images_generate(**kw):
        calls.append(kw)
        return _fake_response(_JPEG_BYTES)

    _run_generate(images_generate, model="gpt-image-1-mini")

    assert calls[0]["model"] == "gpt-image-1-mini"
    assert calls[0]["size"] == "1536x1024"


def test_generate_fallback_image_openai_appends_rendering_instructions_to_prompt():
    calls = []

    def images_generate(**kw):
        calls.append(kw)
        return _fake_response(_JPEG_BYTES)

    _run_generate(images_generate, rendering_instructions="No watermarks.")

    assert "panic pool owner" in calls[0]["prompt"]
    assert "No watermarks." in calls[0]["prompt"]


def test_generate_fallback_image_openai_returns_false_on_non_retryable_error_without_retrying():
    calls = []

    def images_generate(**kw):
        calls.append(kw)
        raise RuntimeError("organization must be verified to use this model")

    success, reason, content, fake_client, _ = _run_generate(images_generate)

    assert success is False
    assert reason is not None and "organization must be verified" in reason
    assert content is None
    assert fake_client.images.generate.call_count == 1  # non-retryable -> no retry


def test_generate_fallback_image_openai_retries_transient_error_then_succeeds():
    attempts = {"n": 0}

    def images_generate(**kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _status_error(503, "temporarily unavailable")
        return _fake_response(_JPEG_BYTES)

    with patch("app.stock.openai_image.asyncio.sleep", return_value=None):
        success, reason, content, fake_client, _ = _run_generate(images_generate, max_attempts=5)

    assert success is True
    assert reason is None
    assert content == _JPEG_BYTES
    assert fake_client.images.generate.call_count == 3


@pytest.mark.parametrize("exc_factory", [
    lambda: _status_error(429, "rate limited"),
    lambda: RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None),
    lambda: APIConnectionError(request=_REQUEST),
    lambda: APITimeoutError(request=_REQUEST),
    lambda: _status_error(500, "server error"),
    lambda: _status_error(502, "bad gateway"),
])
def test_generate_fallback_image_openai_treats_transient_errors_as_retryable(exc_factory):
    assert openai_image._is_retryable(exc_factory()) is True


@pytest.mark.parametrize("exc_factory", [
    lambda: RuntimeError("boom"),
    lambda: _status_error(400, "bad request"),
    lambda: _status_error(401, "unauthorized"),
    lambda: ValueError("bad response shape"),
])
def test_generate_fallback_image_openai_treats_non_transient_errors_as_non_retryable(exc_factory):
    assert openai_image._is_retryable(exc_factory()) is False


def test_generate_fallback_image_openai_exhausts_retries_and_reports_failure():
    def images_generate(**kw):
        raise _status_error(503, "still down")

    with patch("app.stock.openai_image.asyncio.sleep", return_value=None):
        success, reason, content, fake_client, _ = _run_generate(images_generate, max_attempts=3)

    assert success is False
    assert reason is not None and "still down" in reason
    assert fake_client.images.generate.call_count == 3


def test_generate_fallback_image_openai_rejects_bytes_not_matching_declared_format():
    def images_generate(**kw):
        return _fake_response(b"not actually a jpeg")  # missing JPEG magic bytes

    success, reason, content, _, leftover_tmp = _run_generate(images_generate)

    assert success is False
    assert reason is not None and "not a valid JPEG" in reason
    assert content is None
    assert leftover_tmp == []  # never partially written


def test_generate_fallback_image_openai_rejects_empty_prompt():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "generated.jpg"
        success, reason = asyncio.run(generate_fallback_image_openai("   ", output_path))
    assert success is False
    assert reason == "prompt must not be empty"


def test_generate_fallback_image_openai_rejects_invalid_quality():
    def images_generate(**kw):
        raise AssertionError("must not call the API for an invalid input")

    success, reason, content, fake_client, _ = _run_generate(images_generate, quality="ultra")

    assert success is False
    assert reason is not None and "quality" in reason
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_rejects_output_path_with_unsupported_suffix():
    def images_generate(**kw):
        raise AssertionError("must not call the API for an invalid input")

    success, reason, content, fake_client, _ = _run_generate(images_generate, output_suffix=".gif")

    assert success is False
    assert reason is not None and "output_path" in reason
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_rejects_output_format_mismatched_with_suffix():
    def images_generate(**kw):
        raise AssertionError("must not call the API for an invalid input")

    success, reason, content, fake_client, _ = _run_generate(
        images_generate, output_suffix=".jpg", output_format="png",
    )

    assert success is False
    assert reason is not None and "does not match" in reason
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_rejects_output_compression_for_png():
    def images_generate(**kw):
        raise AssertionError("must not call the API for an invalid input")

    success, reason, content, fake_client, _ = _run_generate(
        images_generate, output_suffix=".png", output_compression=80,
    )

    assert success is False
    assert reason is not None and "output_compression" in reason
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_rejects_out_of_range_output_compression():
    def images_generate(**kw):
        raise AssertionError("must not call the API for an invalid input")

    success, reason, content, fake_client, _ = _run_generate(
        images_generate, output_suffix=".webp", output_compression=150,
    )

    assert success is False
    assert reason is not None and "output_compression" in reason
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_accepts_valid_output_compression():
    calls = []

    def images_generate(**kw):
        calls.append(kw)
        return _fake_response(_WEBP_BYTES)

    success, reason, content, _, _ = _run_generate(
        images_generate, output_suffix=".webp", output_compression=80,
    )

    assert success is True
    assert calls[0]["output_compression"] == 80


def test_generate_fallback_image_openai_rejects_max_attempts_below_one():
    success, reason, content, fake_client, _ = _run_generate(
        lambda **kw: _fake_response(_JPEG_BYTES), max_attempts=0,
    )
    assert success is False
    assert reason == "max_attempts must be at least 1"
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_rejects_non_positive_timeout():
    success, reason, content, fake_client, _ = _run_generate(
        lambda **kw: _fake_response(_JPEG_BYTES), request_timeout_seconds=0,
    )
    assert success is False
    assert reason == "request_timeout_seconds must be greater than 0"
    assert fake_client.images.generate.call_count == 0


def test_generate_fallback_image_openai_propagates_cancellation():
    def images_generate(**kw):
        raise asyncio.CancelledError()

    fake_client = MagicMock()
    fake_client.images.generate.side_effect = images_generate
    with patch.object(openai_image, "_client", return_value=fake_client):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.jpg"
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(generate_fallback_image_openai("panic pool owner", output_path))


if __name__ == "__main__":
    test_generate_fallback_image_openai_writes_file_and_returns_true_on_success()
    test_generate_fallback_image_openai_passes_configured_quality_and_default_size()
    test_generate_fallback_image_openai_uses_legacy_size_for_older_model()
    test_generate_fallback_image_openai_appends_rendering_instructions_to_prompt()
    test_generate_fallback_image_openai_returns_false_on_non_retryable_error_without_retrying()
    test_generate_fallback_image_openai_retries_transient_error_then_succeeds()
    test_generate_fallback_image_openai_exhausts_retries_and_reports_failure()
    test_generate_fallback_image_openai_rejects_bytes_not_matching_declared_format()
    test_generate_fallback_image_openai_rejects_empty_prompt()
    test_generate_fallback_image_openai_rejects_invalid_quality()
    test_generate_fallback_image_openai_rejects_output_path_with_unsupported_suffix()
    test_generate_fallback_image_openai_rejects_output_format_mismatched_with_suffix()
    test_generate_fallback_image_openai_rejects_output_compression_for_png()
    test_generate_fallback_image_openai_rejects_out_of_range_output_compression()
    test_generate_fallback_image_openai_accepts_valid_output_compression()
    test_generate_fallback_image_openai_rejects_max_attempts_below_one()
    test_generate_fallback_image_openai_rejects_non_positive_timeout()
    print("OK")
