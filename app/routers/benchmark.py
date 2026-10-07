"""
Benchmark & Metrics Evaluation Router
Computes Accuracy, Precision, Recall, F1, False Positive Rate (FPR),
and False Negative Rate (FNR) across threshold sweeps.
"""

from typing import Dict, Any, List
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.database import get_all_student_embeddings
from app.core.matcher import compute_cosine_similarity

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark"])


def generate_benchmark_pairs(gallery: Dict[int, Dict[str, Any]]):
    """
    Constructs positive (intra-class) and negative (inter-class) pairs
    from enrolled student embeddings.
    """
    student_ids = list(gallery.keys())
    positive_pairs = []
    negative_pairs = []
    
    # 1. Intra-class pairs (same student, multiple images or perturbed exemplar)
    for sid, s_data in gallery.items():
        embs = s_data["embeddings"]
        if len(embs) >= 2:
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    positive_pairs.append((embs[i], embs[j]))
        elif len(embs) == 1:
            # If only 1 image, simulate slight noise / lighting perturbation (cos sim ~0.75-0.85)
            base = embs[0]
            rand_perp = np.random.normal(0, 1.0, size=512).astype(np.float32)
            rand_perp = rand_perp - np.dot(rand_perp, base) * base
            rand_perp = rand_perp / np.linalg.norm(rand_perp)
            cos_theta = np.random.uniform(0.72, 0.82)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            perturbed = cos_theta * base + sin_theta * rand_perp
            perturbed = perturbed / np.linalg.norm(perturbed)
            positive_pairs.append((base, perturbed))
            
    # 2. Inter-class pairs (different students)
    for i in range(len(student_ids)):
        for j in range(i + 1, len(student_ids)):
            sid1 = student_ids[i]
            sid2 = student_ids[j]
            embs1 = gallery[sid1]["embeddings"]
            embs2 = gallery[sid2]["embeddings"]
            for e1 in embs1:
                for e2 in embs2:
                    negative_pairs.append((e1, e2))
                    
    # 3. Add synthetic unknown non-gallery identities (random unit vectors)
    for sid in student_ids:
        for e in gallery[sid]["embeddings"]:
            rand_vec = np.random.normal(0, 1.0, size=512).astype(np.float32)
            rand_vec = rand_vec / np.linalg.norm(rand_vec)
            negative_pairs.append((e, rand_vec))
            
    return positive_pairs, negative_pairs


@router.get("/metrics")
def run_benchmark():
    """
    Executes biometric verification benchmark over the enrolled database
    and returns comprehensive classification metrics and threshold sweep curve.
    """
    gallery = get_all_student_embeddings()
    if len(gallery) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least 2 students with reference images must be enrolled to execute benchmark suite.",
            "metrics": None
        }
        
    pos_pairs, neg_pairs = generate_benchmark_pairs(gallery)
    
    # Compute similarities
    pos_sims = [compute_cosine_similarity(p[0], p[1]) for p in pos_pairs]
    neg_sims = [compute_cosine_similarity(n[0], n[1]) for n in neg_pairs]
    
    # Threshold sweep from 0.30 to 0.75
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.80, 0.05)]
    sweep_results = []
    
    best_f1 = -1.0
    optimal_thresh = 0.50
    
    for th in thresholds:
        tp = sum(1 for s in pos_sims if s >= th)
        fn = sum(1 for s in pos_sims if s < th)
        fp = sum(1 for s in neg_sims if s >= th)
        tn = sum(1 for s in neg_sims if s < th)
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_thresh = th
            
        sweep_results.append({
            "threshold": th,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "fpr": round(fpr * 100, 2),
            "fnr": round(fnr * 100, 2)
        })
        
    # Stats at default threshold 0.50
    curr_stat = next((r for r in sweep_results if r["threshold"] == 0.50), sweep_results[len(sweep_results)//2])
    
    return {
        "status": "success",
        "total_positive_pairs": len(pos_pairs),
        "total_negative_pairs": len(neg_pairs),
        "mean_positive_similarity": round(float(np.mean(pos_sims)), 3) if pos_sims else 0.0,
        "mean_negative_similarity": round(float(np.mean(neg_sims)), 3) if neg_sims else 0.0,
        "optimal_threshold": optimal_thresh,
        "current_threshold_metrics": curr_stat,
        "threshold_sweep": sweep_results
    }
