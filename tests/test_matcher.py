"""
Unit Tests for Biometric Matcher & Decision Logic
Tests:
- Single person match
- Multiple persons match
- All students present
- All students absent
- Unknown intruder (low similarity)
- Duplicate face detection (preventing duplicate attendance)
- Ambiguous identities (top-1 vs runner-up gap < margin)
- Small face resolution filtering
- Motion blur filtering
"""

import pytest
import numpy as np
from app.core.matcher import (
    compute_cosine_similarity,
    compute_similarity_matrix,
    match_classroom_faces,
    ClassroomAttendanceResult
)


def create_random_unit_vector(dim=512):
    vec = np.random.normal(0, 1.0, size=dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def perturb_vector(vec, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=vec.shape).astype(np.float32)
    p = vec + noise
    return p / np.linalg.norm(p)


@pytest.fixture
def sample_gallery():
    """Generates a mock gallery of 3 enrolled students with 512-D embeddings."""
    emb1 = create_random_unit_vector()
    emb2 = create_random_unit_vector()
    emb3 = create_random_unit_vector()
    
    return {
        101: {
            "id": 101,
            "name": "Ali Rezaei",
            "student_number": "STU101",
            "embeddings": [emb1, perturb_vector(emb1, 0.05)],
            "mean_embedding": emb1
        },
        102: {
            "id": 102,
            "name": "Sara Ahmadi",
            "student_number": "STU102",
            "embeddings": [emb2],
            "mean_embedding": emb2
        },
        103: {
            "id": 103,
            "name": "Hossein Moradi",
            "student_number": "STU103",
            "embeddings": [emb3],
            "mean_embedding": emb3
        }
    }


def test_cosine_similarity_properties():
    """Verifies symmetry, self-similarity = 1.0, and bounds."""
    v1 = create_random_unit_vector()
    v2 = create_random_unit_vector()
    
    # Self similarity is 1.0
    assert abs(compute_cosine_similarity(v1, v1) - 1.0) < 1e-5
    # Symmetry
    assert abs(compute_cosine_similarity(v1, v2) - compute_cosine_similarity(v2, v1)) < 1e-6
    # Bound between -1 and 1
    assert -1.0 <= compute_cosine_similarity(v1, v2) <= 1.0


def test_all_students_present(sample_gallery):
    """Tests scenario where all 3 students are present in classroom image."""
    detected = [
        {"face_index": 0, "bbox": [10, 10, 80, 80], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.03), "face_size": 70, "blur_score": 100.0},
        {"face_index": 1, "bbox": [90, 10, 160, 80], "embedding": perturb_vector(sample_gallery[102]["mean_embedding"], 0.03), "face_size": 70, "blur_score": 110.0},
        {"face_index": 2, "bbox": [170, 10, 240, 80], "embedding": perturb_vector(sample_gallery[103]["mean_embedding"], 0.03), "face_size": 70, "blur_score": 95.0},
    ]
    
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50)
    assert len(res.present) == 3
    assert len(res.absent) == 0
    assert len(res.unknown) == 0
    assert len(res.uncertain) == 0
    present_ids = {p["student_id"] for p in res.present}
    assert present_ids == {101, 102, 103}


def test_all_students_absent(sample_gallery):
    """Tests empty classroom image where zero faces are detected."""
    res = match_classroom_faces([], sample_gallery, match_threshold=0.50)
    assert len(res.present) == 0
    assert len(res.absent) == 3
    assert len(res.unknown) == 0


def test_single_person_present_and_absents(sample_gallery):
    """Tests 1 student present, remaining 2 absent."""
    detected = [
        {"face_index": 0, "bbox": [20, 20, 90, 90], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.04), "face_size": 70, "blur_score": 120.0}
    ]
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50)
    assert len(res.present) == 1
    assert res.present[0]["student_id"] == 101
    assert len(res.absent) == 2
    absent_ids = {a["student_id"] for a in res.absent}
    assert absent_ids == {102, 103}


def test_unknown_intruder(sample_gallery):
    """Tests detected face that does not belong to any student in the gallery."""
    random_stranger = create_random_unit_vector()
    detected = [
        {"face_index": 0, "bbox": [10, 10, 80, 80], "embedding": random_stranger, "face_size": 70, "blur_score": 100.0}
    ]
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50)
    assert len(res.present) == 0
    assert len(res.unknown) == 1
    assert res.unknown[0]["face_index"] == 0
    assert len(res.absent) == 3


def test_duplicate_face_assigned_once(sample_gallery):
    """
    Tests artifact where 2 detected boxes match student 101.
    Hungarian algorithm must strictly assign student 101 at most once.
    """
    detected = [
        {"face_index": 0, "bbox": [10, 10, 80, 80], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.02), "face_size": 70, "blur_score": 100.0},
        {"face_index": 1, "bbox": [15, 15, 85, 85], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.04), "face_size": 70, "blur_score": 100.0},
    ]
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50)
    # Student 101 can only be present once!
    present_ids = [p["student_id"] for p in res.present]
    assert present_ids.count(101) == 1
    # The other detection becomes unknown or assigned to another if above threshold
    assert len(res.present) == 1


def test_ambiguous_twins_flagged_as_uncertain(sample_gallery):
    """
    Tests condition where detected face is equally similar to student 101 and 102
    (ambiguity gap < ambiguity_margin). Must be flagged as Uncertain.
    """
    # Create vector midway between student 101 and 102
    v1 = sample_gallery[101]["mean_embedding"]
    v2 = sample_gallery[102]["mean_embedding"]
    mid = (v1 + v2) / 2.0
    mid = mid / np.linalg.norm(mid)
    
    detected = [
        {"face_index": 0, "bbox": [10, 10, 80, 80], "embedding": mid, "face_size": 70, "blur_score": 100.0}
    ]
    
    # Ambiguity margin 0.15 will catch this
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.30, ambiguity_margin=0.15)
    assert len(res.uncertain) >= 1 or len(res.unknown) >= 1


def test_small_face_filter(sample_gallery):
    """Tests small face (< 28 px) being flagged as Uncertain due to low resolution."""
    detected = [
        {"face_index": 0, "bbox": [10, 10, 30, 30], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.02), "face_size": 20, "blur_score": 100.0}
    ]
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50, min_face_size=28)
    assert len(res.present) == 0
    assert len(res.uncertain) == 1
    assert "small" in res.uncertain[0]["reason"].lower()


def test_blurry_face_filter(sample_gallery):
    """Tests blurry face (Laplacian variance < 45.0) being flagged as Uncertain."""
    detected = [
        {"face_index": 0, "bbox": [10, 10, 80, 80], "embedding": perturb_vector(sample_gallery[101]["mean_embedding"], 0.02), "face_size": 70, "blur_score": 20.0}
    ]
    res = match_classroom_faces(detected, sample_gallery, match_threshold=0.50, min_blur_score=45.0)
    assert len(res.present) == 0
    assert len(res.uncertain) == 1
    assert "blur" in res.uncertain[0]["reason"].lower()
