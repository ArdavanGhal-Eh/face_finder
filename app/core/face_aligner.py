"""
Face Alignment Module
Performs canonical 5-point landmark similarity transformation (Umeyama algorithm)
to standard 112x112 pixel face crop required by ArcFace models.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# Canonical ArcFace 112x112 5-point facial landmark reference coordinates
ARCFACE_REF_POINTS_112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Computes optimal similarity transformation (rotation, scale, translation, no shear)
    from src_pts to dst_pts using Umeyama algorithm / Least-Squares.
    """
    # cv2.estimateAffinePartial2D solves the 4-DOF similarity transform (tx, ty, scale, angle)
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if transform_matrix is None:
        # Fallback to standard affine
        transform_matrix = cv2.getAffineTransform(src_pts[:3].astype(np.float32), dst_pts[:3].astype(np.float32))
    return transform_matrix


def align_face_112(
    image_bgr: np.ndarray,
    landmarks_5pts: np.ndarray,
    target_size: Tuple[int, int] = (112, 112)
) -> Optional[np.ndarray]:
    """
    Aligns and normalizes face pose using 5 detected facial landmarks.
    
    Args:
        image_bgr: Source image in BGR format
        landmarks_5pts: NumPy array of shape (5, 2) containing [x, y] coordinates
        target_size: Output resolution, default (112, 112)
        
    Returns:
        np.ndarray: Aligned face crop (112, 112, 3) BGR or None if transform fails
    """
    if landmarks_5pts is None or len(landmarks_5pts) != 5:
        return None
        
    src = np.array(landmarks_5pts, dtype=np.float32)
    dst = ARCFACE_REF_POINTS_112
    
    try:
        M = estimate_similarity_transform(src, dst)
        if M is None:
            return None
            
        aligned = cv2.warpAffine(
            image_bgr,
            M,
            target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return aligned
    except Exception:
        return None


def crop_face_fallback(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
    margin_ratio: float = 0.2
) -> np.ndarray:
    """
    Fallback square crop from bounding box when landmark alignment fails.
    """
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    side = max(bw, bh) * (1.0 + margin_ratio)
    
    x1_sq = max(0, int(cx - side / 2.0))
    y1_sq = max(0, int(cy - side / 2.0))
    x2_sq = min(w_img, int(cx + side / 2.0))
    y2_sq = min(h_img, int(cy + side / 2.0))
    
    crop = image_bgr[y1_sq:y2_sq, x1_sq:x2_sq]
    if crop.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
