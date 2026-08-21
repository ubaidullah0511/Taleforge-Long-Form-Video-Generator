from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from app.config import settings
from app.llm_client import _client

logger = logging.getLogger(__name__)

ImageQuality = Literal["low", "medium", "high", "auto"]
ImageFormat = Literal["png", "jpeg", "webp"]

_TRANSIENT_STATUS_CODES = {
    408,
    409,
    429,
    500,
    502,
    503,
    504,
}

_SUFFIX_TO_FORMAT: dict[str, ImageFormat] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
}


def _is_retryable(exc: Exception) -> bool:
    """Return True only for temporary errors worth retrying."""
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ),
    ):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code in _TRANSIENT_STATUS_CODES

    return False


def _format_failure(exc: Exception) -> str:
    """Create a useful failure reason without logging the full prompt."""
    parts = [type(exc).__name__]

    code = getattr(exc, "code", None)
    if code:
        parts.append(f"code={code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        parts.append(f"request_id={request_id}")

    detail = str(exc).strip()
    if detail:
        parts.append(detail[:500])

    return " | ".join(parts)


def _validate_image_bytes(
    data: bytes,
    output_format: ImageFormat,
) -> None:
    """Ensure the decoded response is actually an image."""
    if not data:
        raise ValueError("OpenAI returned an empty image payload")

    if output_format == "png":
        valid = data.startswith(b"\x89PNG\r\n\x1a\n")

    elif output_format == "jpeg":
        valid = data.startswith(b"\xff\xd8\xff")

    else:
        valid = (
            len(data) >= 12
            and data.startswith(b"RIFF")
            and data[8:12] == b"WEBP"
        )

    if not valid:
        raise ValueError(
            f"Decoded response is not a valid "
            f"{output_format.upper()} image"
        )


def _atomic_write(path: Path, data: bytes) -> None:
    """Prevent incomplete images from appearing at the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    finally:
        temporary_path.unlink(missing_ok=True)


async def generate_fallback_image_openai(
    prompt: str,
    output_path: Path,
    size: str | None = None,
    *,
    model: str | None = None,
    quality: ImageQuality | None = None,
    output_format: ImageFormat | None = None,
    output_compression: int | None = None,
    rendering_instructions: str | None = None,
    max_attempts: int = 3,
    request_timeout_seconds: float = 180.0,
) -> tuple[bool, str | None]:
    """Generate one OpenAI image.

    Preserves the existing return contract:

        (True, None)
        (False, failure_reason)

    The cached synchronous OpenAI client runs inside a worker thread,
    keeping the asyncio event loop responsive.
    """
    clean_prompt = prompt.strip()

    if not clean_prompt:
        return False, "prompt must not be empty"

    output_path = Path(output_path)

    resolved_model = model or getattr(
        settings,
        "ai_generation_model",
        "gpt-image-1-mini",
    )

    resolved_quality = quality or getattr(
        settings,
        "ai_generation_quality",
        "medium",
    )

    if resolved_quality not in {
        "low",
        "medium",
        "high",
        "auto",
    }:
        return (
            False,
            "quality must be low, medium, high, or auto; "
            f"received {resolved_quality!r}",
        )

    # GPT Image 2 supports exact 16:9 output.
    # Older GPT Image models use the legacy landscape size.
    resolved_size = size or (
        "2048x1152"
        if resolved_model == "gpt-image-2"
        else "1536x1024"
    )

    inferred_format = _SUFFIX_TO_FORMAT.get(
        output_path.suffix.lower()
    )

    if output_format is None:
        if inferred_format is None:
            return (
                False,
                "output_path must end with "
                ".png, .jpg, .jpeg, or .webp",
            )

        resolved_format = inferred_format

    else:
        resolved_format = output_format

        if inferred_format != resolved_format:
            return (
                False,
                f"Output suffix {output_path.suffix!r} does not "
                f"match output_format={resolved_format!r}",
            )

    if output_compression is not None:
        if resolved_format == "png":
            return (
                False,
                "output_compression is only supported "
                "for JPEG and WebP",
            )

        if not 0 <= output_compression <= 100:
            return (
                False,
                "output_compression must be between 0 and 100",
            )

    if max_attempts < 1:
        return False, "max_attempts must be at least 1"

    if request_timeout_seconds <= 0:
        return (
            False,
            "request_timeout_seconds must be greater than 0",
        )

    final_prompt = clean_prompt

    if rendering_instructions:
        clean_instructions = rendering_instructions.strip()

        if clean_instructions:
            final_prompt = (
                f"{clean_prompt}\n\n"
                "Rendering requirements:\n"
                f"{clean_instructions}"
            )

    # Use a hash for logging instead of exposing the entire prompt.
    prompt_id = hashlib.sha256(
        final_prompt.encode("utf-8")
    ).hexdigest()[:12]

    def _generate() -> tuple[bytes, str | None]:
        request: dict[str, object] = {
            "model": resolved_model,
            "prompt": final_prompt,
            "size": resolved_size,
            "quality": resolved_quality,
            "output_format": resolved_format,
            "n": 1,
            "timeout": request_timeout_seconds,
        }

        if output_compression is not None:
            request["output_compression"] = output_compression

        response = _client().images.generate(**request)

        if not response.data:
            raise ValueError("OpenAI returned no image data")

        encoded_image = response.data[0].b64_json

        if not encoded_image:
            raise ValueError(
                "OpenAI returned image data without b64_json"
            )

        try:
            image_bytes = base64.b64decode(
                encoded_image,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "OpenAI returned invalid base64 image data"
            ) from exc

        _validate_image_bytes(
            image_bytes,
            resolved_format,
        )

        request_id = getattr(
            response,
            "_request_id",
            None,
        )

        return image_bytes, request_id

    for attempt in range(1, max_attempts + 1):
        try:
            image_bytes, request_id = await asyncio.to_thread(
                _generate
            )

            await asyncio.to_thread(
                _atomic_write,
                output_path,
                image_bytes,
            )

            logger.info(
                "openai_image: generated "
                "prompt_id=%s model=%s size=%s quality=%s "
                "format=%s request_id=%s output=%s",
                prompt_id,
                resolved_model,
                resolved_size,
                resolved_quality,
                resolved_format,
                request_id,
                output_path,
            )

            return True, None

        except asyncio.CancelledError:
            # Preserve proper asyncio cancellation behavior.
            raise

        except Exception as exc:
            failure = _format_failure(exc)
            retryable = _is_retryable(exc)

            if not retryable or attempt == max_attempts:
                logger.warning(
                    "openai_image: failed "
                    "prompt_id=%s model=%s attempt=%d/%d "
                    "retryable=%s error=%s",
                    prompt_id,
                    resolved_model,
                    attempt,
                    max_attempts,
                    retryable,
                    failure,
                )

                return False, failure

            # Exponential backoff plus small jitter.
            delay = min(
                8.0,
                0.75 * (2 ** (attempt - 1)),
            )
            delay += random.uniform(0.0, 0.25)

            logger.info(
                "openai_image: retrying "
                "prompt_id=%s attempt=%d/%d in %.2fs "
                "after error=%s",
                prompt_id,
                attempt,
                max_attempts,
                delay,
                failure,
            )

            await asyncio.sleep(delay)

    return False, "image generation failed unexpectedly"
