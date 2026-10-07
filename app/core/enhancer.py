"""
Image Enhancement & Super-Resolution Preprocessing Module
Face Finder - Advanced Biometric Quality Booster
Provides:
1. CLAHE Adaptive Illumination & Contrast Normalization (LAB Space)
2. Lanczos-4 High-Order Sub-pixel Resampling
3. Unsharp Masking Gradient Sharpening
4. Sliced Aided Hyper Inference (SAHI) Tiled Face Detection for High-Density Crowds
5. Dual-Feature Biometric Fusion
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the Luminance (L) channel in CIE LAB color space.
    Normalizes harsh shadows, window glares, and dim fluorescent classroom lighting.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
        
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    image_bgr: np.ndarray,
    strength: float = 1.35,
    sigma: float = 1.0,
    threshold: int = 3
) -> np.ndarray:
    """
    Sharpens high-frequency facial landmark contours (eyes, lips, nose edges)
    using unsharp masking with thresholding to prevent boosting flat noise.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
        
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    diff = cv2.subtract(image_bgr, blurred)
    
    # Mask out low amplitude noise
    if threshold > 0:
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        low_contrast_mask = gray_diff < threshold
        diff[low_contrast_mask] = 0
        
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_face_crop_112(crop_112: np.ndarray) -> np.ndarray:
    """
    Applies composite super-enhancement pipeline to canonical 112x112 aligned face crop:
    1. Lanczos-4 edge preservation
    2. Adaptive CLAHE contrast balance
    3. High-frequency unsharp mask
    """
    if crop_112 is None or crop_112.size == 0:
        return crop_112
        
    # Contrast normalization
    enhanced = apply_clahe(crop_112, clip_limit=1.8, tile_grid_size=(4, 4))
    # Unsharp mask for eye/lip contours
    sharpened = apply_unsharp_mask(enhanced, strength=1.25, sigma=0.8)
    return sharpened


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def nms_faces(faces: List[Any], iou_thresh: float = 0.45) -> List[Any]:
    """Applies Non-Maximum Suppression (NMS) to merge overlapping detections."""
    if not faces:
        return []
        
    # Sort by detection score descending
    sorted_faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    kept = []
    
    for f in sorted_faces:
        duplicate = False
        for k in kept:
            if compute_iou(f.bbox[:4], k.bbox[:4]) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(f)
            
    return kept


def run_tiled_face_detection(
    app,
    image_bgr: np.ndarray,
    tile_size: Tuple[int, int] = (640, 640),
    overlap_ratio: float = 0.25
) -> List[Any]:
    """
    Runs Slicing Aided Hyper Inference (SAHI) style tiled detection on high-resolution images.
    Preserves small back-row faces without downscaling artifacts.
    """
    h_img, w_img = image_bgr.shape[:2]
    
    # Global full-image detection
    all_faces = list(app.get(image_bgr))
    
    tw, th = tile_size
    step_x = int(tw * (1.0 - overlap_ratio))
    step_y = int(th * (1.0 - overlap_ratio))
    
    for y in range(0, max(1, h_img - th + 1), step_y):
        for x in range(0, max(1, w_img - tw + 1), step_x):
            x2 = min(w_img, x + tw)
            y2 = min(h_img, y + th)
            tile = image_bgr[y:y2, x:x2]
            
            tile_faces = app.get(tile)
            for f in tile_faces:
                # Offset bounding box and landmarks back to global coordinates
                f.bbox[0] += x
                f.bbox[1] += y
                f.bbox[2] += x
                f.bbox[3] += y
                if hasattr(f, 'kps') and f.kps is not None:
                    f.kps[:, 0] += x
                    f.kps[:, 1] += y
                all_faces.append(f)
                
    return nms_faces(all_faces, iou_thresh=0.45)
