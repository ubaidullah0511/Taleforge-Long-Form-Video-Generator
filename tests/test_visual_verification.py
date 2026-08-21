"""Offline tests for app.visual_verification — the CLIP model and frame
extraction are mocked out, so no GPU/model download/real video file is
needed. Run with: pytest tests/test_visual_verification.py
"""
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from app.config import settings
from app.visual_verification import passes_visual_verification, visual_match_score


def _fake_model(image_vec, text_vec):
    """A stand-in for clip.load()'s (model, preprocess) pair whose
    encode_image/encode_text return fixed vectors, so the resulting cosine
    similarity is fully controlled by the test."""
    model = MagicMock()
    model.encode_image.return_value = torch.tensor([image_vec])
    model.encode_text.return_value = torch.tensor([text_vec])
    preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    return model, preprocess


def test_visual_match_score_returns_none_when_frame_extraction_fails():
    with patch("app.visual_verification._extract_sample_frame", return_value=None):
        assert visual_match_score("missing.mp4", "a truck on a highway") is None


def test_visual_match_score_high_similarity_for_matching_vectors():
    model, preprocess = _fake_model([1.0, 0.0], [1.0, 0.0])  # identical direction -> similarity 1.0
    with patch("app.visual_verification._extract_sample_frame", return_value=Image.new("RGB", (4, 4))), \
         patch("app.visual_verification._load_clip_model", return_value=(model, preprocess)):
        score = visual_match_score("truck.mp4", "a semi truck on a highway")
    assert score == 1.0


def test_visual_match_score_low_similarity_for_orthogonal_vectors():
    model, preprocess = _fake_model([1.0, 0.0], [0.0, 1.0])  # orthogonal -> similarity 0.0
    with patch("app.visual_verification._extract_sample_frame", return_value=Image.new("RGB", (4, 4))), \
         patch("app.visual_verification._load_clip_model", return_value=(model, preprocess)):
        score = visual_match_score("car.mp4", "a semi truck on a highway")
    assert score == 0.0


def test_passes_visual_verification_frame_extraction_failure_defaults_to_pass():
    """A corrupted/unreadable video file is a different failure mode
    (handled by asset download validation) — it must not be treated as a
    content mismatch by this gate."""
    with patch("app.visual_verification.visual_match_score", return_value=None):
        passed, score = passes_visual_verification("unreadable.mp4", "a truck")
    assert passed is True
    assert score is None


def test_passes_visual_verification_rejects_below_threshold():
    with patch("app.visual_verification.visual_match_score", return_value=0.10), \
         patch.object(settings, "visual_verification_threshold", 0.26):
        passed, score = passes_visual_verification("video.mp4", "a truck")
    assert passed is False
    assert score == 0.10


def test_passes_visual_verification_accepts_at_or_above_threshold():
    with patch("app.visual_verification.visual_match_score", return_value=0.30), \
         patch.object(settings, "visual_verification_threshold", 0.26):
        passed, score = passes_visual_verification("video.mp4", "a truck")
    assert passed is True
    assert score == 0.30


def test_passes_visual_verification_skips_entirely_when_disabled():
    with patch.object(settings, "enable_visual_verification", False), \
         patch("app.visual_verification.visual_match_score") as mock_score:
        passed, score = passes_visual_verification("video.mp4", "a truck")
    assert passed is True
    assert score is None
    mock_score.assert_not_called()


if __name__ == "__main__":
    test_visual_match_score_returns_none_when_frame_extraction_fails()
    test_visual_match_score_high_similarity_for_matching_vectors()
    test_visual_match_score_low_similarity_for_orthogonal_vectors()
    test_passes_visual_verification_frame_extraction_failure_defaults_to_pass()
    test_passes_visual_verification_rejects_below_threshold()
    test_passes_visual_verification_accepts_at_or_above_threshold()
    test_passes_visual_verification_skips_entirely_when_disabled()
    print("OK")
