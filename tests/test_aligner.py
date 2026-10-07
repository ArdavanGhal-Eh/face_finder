"""
Unit Tests for Face Aligner & 5-point Landmark Transformation
"""

import numpy as np
import cv2
from app.core.face_aligner import (
    align_face_112,
    estimate_similarity_transform,
    crop_face_fallback,
    ARCFACE_REF_POINTS_112
)


def test_similarity_transform_exact_match():
    """Checks that identical landmarks yield an identity transformation."""
    pts = ARCFACE_REF_POINTS_112.copy()
    M = estimate_similarity_transform(pts, pts)
    assert M is not None
    # Diagonal should be ~1.0, translation ~0
    np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-3)
    np.testing.assert_allclose(M[:, 2], np.zeros(2), atol=1e-3)


def test_align_face_shape_and_type():
    """Verifies that align_face_112 produces standard (112, 112, 3) image output."""
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks = ARCFACE_REF_POINTS_112 + 50.0  # shifted
    aligned = align_face_112(dummy_img, landmarks)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_crop_face_fallback():
    """Verifies fallback square cropping."""
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    bbox = np.array([40, 40, 100, 100])
    crop = crop_face_fallback(dummy_img, bbox, target_size=(112, 112))
    assert crop.shape == (112, 112, 3)
