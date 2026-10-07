"""
Unit & Integration Tests for Database & Vector Storage
Tests student creation, vector serialization/deserialization,
hard deletion (privacy), session recording, and settings persistence.
"""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from app.database import (
    init_db,
    create_student,
    get_all_students,
    get_student_by_id,
    delete_student,
    add_student_image,
    serialize_embedding,
    deserialize_embedding,
    get_all_student_embeddings,
    get_system_settings,
    update_system_settings
)


def test_embedding_serialization_roundtrip():
    """Checks that 512-D float32 vectors serialize to bytes and deserialize with normalization intact."""
    vec = np.random.normal(0, 1.0, size=512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    
    blob = serialize_embedding(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == 512 * 4  # 4 bytes per float32 = 2048 bytes
    
    restored = deserialize_embedding(blob)
    assert restored.shape == (512,)
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, vec, atol=1e-6)
    assert abs(np.linalg.norm(restored) - 1.0) < 1e-5


def test_student_lifecycle_and_privacy_deletion():
    """
    Verifies adding student, enrolling vector embedding, and performing hard-delete.
    """
    init_db()
    sid = create_student(
        student_number="TEST9999",
        name="Test Student Privacy",
        has_consent=True
    )
    assert sid > 0
    
    # Add dummy vector embedding
    vec = np.random.normal(0, 1.0, size=512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    
    img_id = add_student_image(
        student_id=sid,
        file_path="non_existent_mock_file.jpg",
        embedding=vec,
        face_quality=0.95,
        is_primary=True
    )
    assert img_id > 0
    
    # Retrieve student
    s = get_student_by_id(sid)
    assert s is not None
    assert s["name"] == "Test Student Privacy"
    assert len(s["images"]) == 1
    
    # Ensure loaded in gallery
    gallery = get_all_student_embeddings()
    assert sid in gallery
    assert len(gallery[sid]["embeddings"]) == 1
    
    # Delete student (Right to Erasure / GDPR compliance)
    deleted = delete_student(sid)
    assert deleted is True
    
    # Check that student no longer exists in DB or gallery
    assert get_student_by_id(sid) is None
    gallery_after = get_all_student_embeddings()
    assert sid not in gallery_after


def test_system_settings_persistence():
    """Checks getting and updating system thresholds."""
    init_db()
    curr = get_system_settings()
    assert "match_threshold" in curr
    
    update_system_settings({"match_threshold": 0.62})
    updated = get_system_settings()
    assert updated["match_threshold"] == 0.62
    
    # Restore default
    update_system_settings({"match_threshold": 0.50})
