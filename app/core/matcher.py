"""
Biometric Matcher Module
Implements Cosine Similarity, Multi-Exemplar Gallery Matching,
Global Optimal Bipartite Matching (Hungarian Algorithm via Scipy),
and Dual-Threshold Identity Decision Logic (Present / Uncertain / Unknown / Absent).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchDecision:
    """Represents decision for an individual detected face."""
    face_index: int
    bbox: List[int]
    status: str  # 'present', 'uncertain', 'unknown'
    assigned_student_id: Optional[int] = None
    student_name: Optional[str] = None
    student_number: Optional[str] = None
    similarity: float = 0.0
    runner_up_similarity: float = 0.0
    runner_up_name: Optional[str] = None
    ambiguity_gap: float = 0.0
    reason: str = ""
    is_blurry: bool = False
    blur_score: float = 0.0
    face_size: int = 0


@dataclass
class ClassroomAttendanceResult:
    """Comprehensive structured attendance evaluation result."""
    present: List[Dict[str, Any]] = field(default_factory=list)
    absent: List[Dict[str, Any]] = field(default_factory=list)
    uncertain: List[Dict[str, Any]] = field(default_factory=list)
    unknown: List[Dict[str, Any]] = field(default_factory=list)
    all_decisions: List[MatchDecision] = field(default_factory=list)
    total_detected: int = 0
    match_threshold_used: float = 0.50
    ambiguity_margin_used: float = 0.08


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Computes cosine similarity between two normalized 1D embedding vectors."""
    dot = float(np.dot(emb1, emb2))
    return max(-1.0, min(1.0, dot))


def compute_similarity_matrix(
    detected_embeddings: List[np.ndarray],
    gallery: Dict[int, Dict[str, Any]]
) -> Tuple[np.ndarray, List[int]]:
    """
    Computes an (M x N) similarity matrix between M detected faces and N gallery students.
    
    Uses a hybrid metric:
    sim(i, j) = 0.6 * max_exemplar_sim(i, j) + 0.4 * mean_embedding_sim(i, j)
    This combines sensitivity to varied pose/lighting exemplars with stability against outliers.
    
    Returns:
        similarity_matrix: np.ndarray of shape (M, N)
        student_ids: List of student IDs corresponding to columns
    """
    M = len(detected_embeddings)
    student_ids = list(gallery.keys())
    N = len(student_ids)
    
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32), student_ids
        
    sim_matrix = np.zeros((M, N), dtype=np.float32)
    
    for i, det_emb in enumerate(detected_embeddings):
        for j, sid in enumerate(student_ids):
            student_data = gallery[sid]
            
            # Exemplar similarities
            exemplar_sims = [
                compute_cosine_similarity(det_emb, ex_emb)
                for ex_emb in student_data["embeddings"]
            ]
            max_sim = max(exemplar_sims) if exemplar_sims else 0.0
            
            # Mean embedding similarity
            mean_sim = 0.0
            if student_data["mean_embedding"] is not None:
                mean_sim = compute_cosine_similarity(det_emb, student_data["mean_embedding"])
                
            # Hybrid score
            sim_matrix[i, j] = 0.6 * max_sim + 0.4 * mean_sim
            
    return sim_matrix, student_ids


def match_classroom_faces(
    detected_faces: List[Dict[str, Any]],
    gallery: Dict[int, Dict[str, Any]],
    match_threshold: float = 0.50,
    ambiguity_margin: float = 0.08,
    min_face_size: int = 28,
    min_blur_score: float = 45.0
) -> ClassroomAttendanceResult:
    """
    Matches detected faces in a classroom photo to enrolled students using the Hungarian algorithm
    and calibrated dual-threshold logic.
    
    Args:
        detected_faces: List of dicts with 'embedding', 'bbox', 'blur_score', 'face_size'
        gallery: Enrolled student gallery from database
        match_threshold: Threshold above which a match is considered positive
        ambiguity_margin: Required separation from runner-up match
        min_face_size: Minimum width/height in pixels
        min_blur_score: Minimum Laplacian variance for sharpness
        
    Returns:
        ClassroomAttendanceResult containing present, absent, uncertain, and unknown groups
    """
    M = len(detected_faces)
    student_ids = list(gallery.keys())
    N = len(student_ids)
    
    result = ClassroomAttendanceResult(
        total_detected=M,
        match_threshold_used=match_threshold,
        ambiguity_margin_used=ambiguity_margin
    )
    
    if N == 0:
        # No students enrolled in database; all detected faces are Unknown
        for i, face in enumerate(detected_faces):
            dec = MatchDecision(
                face_index=i,
                bbox=face["bbox"],
                status="unknown",
                similarity=0.0,
                reason="No enrolled students in database",
                is_blurry=face.get("blur_score", 100.0) < min_blur_score,
                blur_score=face.get("blur_score", 100.0),
                face_size=face.get("face_size", 0)
            )
            result.all_decisions.append(dec)
            result.unknown.append({
                "face_index": i,
                "bbox": face["bbox"],
                "best_similarity": 0.0,
                "best_candidate_name": None,
                "reason": dec.reason,
                "blur_score": dec.blur_score,
                "face_size": dec.face_size
            })
        return result
        
    if M == 0:
        # No faces detected; all active students are Absent
        for sid in student_ids:
            s_data = gallery[sid]
            result.absent.append({
                "student_id": sid,
                "name": s_data["name"],
                "student_number": s_data["student_number"]
            })
        return result
        
    # Extract embeddings
    det_embs = [f["embedding"] for f in detected_faces]
    sim_matrix, col_student_ids = compute_similarity_matrix(det_embs, gallery)
    
    # Global Optimal Bipartite Matching (Hungarian Algorithm)
    # Convert similarity matrix to cost matrix: Cost = 1.0 - Similarity
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Map matched rows and track student assignments
    assigned_matches: Dict[int, int] = {}  # face_index -> col_index
    assigned_students: set = set()          # student_id
    
    for r, c in zip(row_ind, col_ind):
        assigned_matches[r] = c
        
    # Evaluate each detected face
    for i in range(M):
        face = detected_faces[i]
        bbox = face["bbox"]
        blur_score = face.get("blur_score", 100.0)
        face_size = face.get("face_size", 50)
        is_blurry = blur_score < min_blur_score
        is_too_small = face_size < min_face_size
        
        # Sort all similarities for face i to find top-1 and runner-up
        face_sims = sim_matrix[i, :]
        sorted_col_indices = np.argsort(face_sims)[::-1]
        
        best_col = sorted_col_indices[0]
        top_sim = float(face_sims[best_col])
        best_sid = col_student_ids[best_col]
        best_student = gallery[best_sid]
        
        runner_up_sim = float(face_sims[sorted_col_indices[1]]) if N > 1 else 0.0
        runner_up_sid = col_student_ids[sorted_col_indices[1]] if N > 1 else None
        runner_up_name = gallery[runner_up_sid]["name"] if runner_up_sid else None
        ambiguity_gap = top_sim - runner_up_sim
        
        # Check Hungarian assignment for this face
        assigned_col = assigned_matches.get(i)
        assigned_sim = float(face_sims[assigned_col]) if assigned_col is not None else -1.0
        
        if assigned_col is not None and assigned_sim >= match_threshold:
            matched_sid = col_student_ids[assigned_col]
            matched_student = gallery[matched_sid]
            
            # 1. Quality Checks
            if is_too_small:
                dec = MatchDecision(
                    face_index=i,
                    bbox=bbox,
                    status="uncertain",
                    assigned_student_id=matched_sid,
                    student_name=matched_student["name"],
                    student_number=matched_student["student_number"],
                    similarity=assigned_sim,
                    runner_up_similarity=runner_up_sim,
                    runner_up_name=runner_up_name,
                    ambiguity_gap=ambiguity_gap,
                    reason=f"Face size too small ({face_size}px < {min_face_size}px) for verified identification",
                    is_blurry=is_blurry,
                    blur_score=blur_score,
                    face_size=face_size
                )
                result.all_decisions.append(dec)
                result.uncertain.append({
                    "face_index": i,
                    "student_id": matched_sid,
                    "name": matched_student["name"],
                    "student_number": matched_student["student_number"],
                    "confidence": round(assigned_sim, 3),
                    "bbox": bbox,
                    "reason": dec.reason
                })
            elif is_blurry:
                dec = MatchDecision(
                    face_index=i,
                    bbox=bbox,
                    status="uncertain",
                    assigned_student_id=matched_sid,
                    student_name=matched_student["name"],
                    student_number=matched_student["student_number"],
                    similarity=assigned_sim,
                    runner_up_similarity=runner_up_sim,
                    runner_up_name=runner_up_name,
                    ambiguity_gap=ambiguity_gap,
                    reason=f"Motion blur detected (Laplacian score {blur_score:.1f} < {min_blur_score})",
                    is_blurry=is_blurry,
                    blur_score=blur_score,
                    face_size=face_size
                )
                result.all_decisions.append(dec)
                result.uncertain.append({
                    "face_index": i,
                    "student_id": matched_sid,
                    "name": matched_student["name"],
                    "student_number": matched_student["student_number"],
                    "confidence": round(assigned_sim, 3),
                    "bbox": bbox,
                    "reason": dec.reason
                })
            # 2. Ambiguity Margin Check
            elif N > 1 and ambiguity_gap < ambiguity_margin and runner_up_sim >= (match_threshold - 0.05):
                dec = MatchDecision(
                    face_index=i,
                    bbox=bbox,
                    status="uncertain",
                    assigned_student_id=matched_sid,
                    student_name=matched_student["name"],
                    student_number=matched_student["student_number"],
                    similarity=assigned_sim,
                    runner_up_similarity=runner_up_sim,
                    runner_up_name=runner_up_name,
                    ambiguity_gap=ambiguity_gap,
                    reason=f"Ambiguous identity: Runner-up '{runner_up_name}' too close (Gap: {ambiguity_gap:.3f} < {ambiguity_margin:.2f})",
                    is_blurry=is_blurry,
                    blur_score=blur_score,
                    face_size=face_size
                )
                result.all_decisions.append(dec)
                result.uncertain.append({
                    "face_index": i,
                    "student_id": matched_sid,
                    "name": matched_student["name"],
                    "student_number": matched_student["student_number"],
                    "confidence": round(assigned_sim, 3),
                    "bbox": bbox,
                    "reason": dec.reason
                })
            else:
                # 3. Confident Match -> Present!
                assigned_students.add(matched_sid)
                dec = MatchDecision(
                    face_index=i,
                    bbox=bbox,
                    status="present",
                    assigned_student_id=matched_sid,
                    student_name=matched_student["name"],
                    student_number=matched_student["student_number"],
                    similarity=assigned_sim,
                    runner_up_similarity=runner_up_sim,
                    runner_up_name=runner_up_name,
                    ambiguity_gap=ambiguity_gap,
                    reason="Confirmed high-confidence biometric match",
                    is_blurry=is_blurry,
                    blur_score=blur_score,
                    face_size=face_size
                )
                result.all_decisions.append(dec)
                result.present.append({
                    "face_index": i,
                    "student_id": matched_sid,
                    "name": matched_student["name"],
                    "student_number": matched_student["student_number"],
                    "confidence": round(assigned_sim, 3),
                    "bbox": bbox,
                    "reason": dec.reason
                })
        else:
            # Below match threshold -> Unknown person
            reason = f"Similarity ({top_sim:.3f}) below threshold ({match_threshold:.2f})"
            if best_student:
                reason += f" (Closest: {best_student['name']})"
            dec = MatchDecision(
                face_index=i,
                bbox=bbox,
                status="unknown",
                similarity=top_sim,
                runner_up_similarity=runner_up_sim,
                runner_up_name=runner_up_name,
                ambiguity_gap=ambiguity_gap,
                reason=reason,
                is_blurry=is_blurry,
                blur_score=blur_score,
                face_size=face_size
            )
            result.all_decisions.append(dec)
            result.unknown.append({
                "face_index": i,
                "bbox": bbox,
                "best_similarity": round(top_sim, 3),
                "best_candidate_name": best_student["name"] if best_student else None,
                "reason": reason,
                "blur_score": blur_score,
                "face_size": face_size
            })
            
    # Compile Absent students: any enrolled student not marked as Present
    for sid in student_ids:
        if sid not in assigned_students:
            s_data = gallery[sid]
            result.absent.append({
                "student_id": sid,
                "name": s_data["name"],
                "student_number": s_data["student_number"]
            })
            
    return result
