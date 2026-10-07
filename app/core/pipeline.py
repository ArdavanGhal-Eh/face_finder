"""
Unified Face Pipeline Orchestrator
Handles SCRFD detection, 5-point alignment, ArcFace 512-D embedding extraction,
blur and quality evaluation, annotation rendering, and classroom attendance execution.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading

import insightface
from insightface.app import FaceAnalysis

from app.config import settings, SESSIONS_DIR, STUDENTS_DIR
from app.core.matcher import (
    match_classroom_faces,
    ClassroomAttendanceResult,
    MatchDecision
)
from app.core.face_aligner import align_face_112


class FacePipeline:
    """Singleton wrapper around InsightFace models and processing utilities."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, model_name: str = "buffalo_s"):
        self.model_name = model_name
        self.app = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initializes InsightFace with detection and recognition modules only."""
        # Use CPUExecutionProvider by default for universal portability
        providers = ['CPUExecutionProvider']
        
        # Check for CUDA or DirectML if available in ONNX Runtime
        import onnxruntime
        available = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.insert(0, 'CUDAExecutionProvider')
        elif 'DirectMLExecutionProvider' in available:
            providers.insert(0, 'DirectMLExecutionProvider')
            
        self.app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        self.app.prepare(
            ctx_id=0,
            det_size=(settings.det_size_w, settings.det_size_h),
            det_thresh=settings.min_det_confidence
        )

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "FacePipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_name=model_name or settings.model_name)
            elif model_name and cls._instance.model_name != model_name:
                cls._instance = cls(model_name=model_name)
            return cls._instance

    @staticmethod
    def calculate_blur_score(face_crop: np.ndarray) -> float:
        """Computes Variance of Laplacian on grayscale face crop to estimate sharpness/blur."""
        if face_crop is None or face_crop.size == 0:
            return 0.0
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def extract_face_for_enrollment(
        self,
        image_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[str]]:
        """
        Processes a single reference image for student enrollment.
        Validates that exactly one prominent face is present.
        
        Returns:
            embedding: Normalized 512-D float32 array or None
            aligned_crop: 112x112 BGR face crop or None
            quality_score: composite quality metric
            error_message: None if successful, otherwise description
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, None, 0.0, "Invalid image data"
            
        faces = self.app.get(image_bgr)
        if len(faces) == 0:
            return None, None, 0.0, "No face detected in the reference image. Please provide a clear, well-lit photo."
            
        if len(faces) > 1:
            # Sort by bounding box area to check if there is one dominant face
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            area_main = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
            area_second = (faces[1].bbox[2] - faces[1].bbox[0]) * (faces[1].bbox[3] - faces[1].bbox[1])
            if area_second > 0.4 * area_main:
                return None, None, 0.0, f"Multiple faces ({len(faces)}) detected. Reference images must contain only the student's face."
                
        face = faces[0]
        x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
        h, w = image_bgr.shape[:2]
        crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        blur_score = self.calculate_blur_score(crop)
        if blur_score < 25.0:
            return None, None, blur_score, f"Image is too blurry (Blur Score: {blur_score:.1f}). Please capture a sharper photo."
            
        # Get normalized embedding
        emb = face.normed_embedding
        if emb is None or len(emb) == 0:
            return None, None, 0.0, "Failed to extract biometric embedding from face."
            
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
            
        # Aligned crop
        aligned = align_face_112(image_bgr, face.kps)
        if aligned is None and crop.size > 0:
            aligned = cv2.resize(crop, (112, 112))
            
        quality = min(1.0, (face.det_score * 0.5) + (min(blur_score, 150.0) / 150.0 * 0.5))
        return emb, aligned, quality, None

    def process_classroom_image(
        self,
        classroom_image_bgr: np.ndarray,
        gallery: Dict[int, Dict[str, Any]],
        match_threshold: float = 0.50,
        ambiguity_margin: float = 0.08,
        min_face_size: int = 28,
        min_blur_score: float = 45.0
    ) -> Tuple[ClassroomAttendanceResult, np.ndarray, List[Dict[str, Any]]]:
        """
        Detects all faces in a group classroom photo, extracts embeddings, matches with gallery,
        annotates the image, and crops unknown/uncertain faces for manual verification.
        
        Returns:
            result: ClassroomAttendanceResult dataclass
            annotated_image_bgr: Image with bounding boxes and name labels
            unknown_crops: List of crop dictionaries with file path and metadata
        """
        h_img, w_img = classroom_image_bgr.shape[:2]
        
        # 1. Run detection & recognition
        raw_faces = self.app.get(classroom_image_bgr)
        
        detected_faces = []
        for i, f in enumerate(raw_faces):
            x1, y1, x2, y2 = [int(v) for v in f.bbox[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            bw = x2 - x1
            bh = y2 - y1
            face_size = min(bw, bh)
            
            crop = classroom_image_bgr[y1:y2, x1:x2]
            blur_score = self.calculate_blur_score(crop)
            
            emb = f.normed_embedding
            if emb is not None and len(emb) > 0:
                emb = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
            else:
                emb = np.zeros(512, dtype=np.float32)
                
            detected_faces.append({
                "face_index": i,
                "bbox": [x1, y1, x2, y2],
                "embedding": emb,
                "det_score": float(f.det_score),
                "blur_score": blur_score,
                "face_size": face_size,
                "kps": f.kps if hasattr(f, 'kps') else None,
                "crop": crop
            })
            
        # 2. Match with student gallery
        attendance_result = match_classroom_faces(
            detected_faces=detected_faces,
            gallery=gallery,
            match_threshold=match_threshold,
            ambiguity_margin=ambiguity_margin,
            min_face_size=min_face_size,
            min_blur_score=min_blur_score
        )
        
        # 3. Render annotations on copy of image
        annotated_image = classroom_image_bgr.copy()
        unknown_crops_info = []
        
        for dec in attendance_result.all_decisions:
            x1, y1, x2, y2 = dec.bbox
            
            # Determine color scheme and label based on status
            if dec.status == "present":
                # Emerald Green
                box_color = (46, 204, 113)  # BGR
                label = f"{dec.student_name} ({dec.similarity:.2f})"
                sub_label = f"ID: {dec.student_number}"
            elif dec.status == "uncertain":
                # Amber / Orange
                box_color = (39, 174, 96) if dec.similarity >= match_threshold else (0, 165, 255)
                box_color = (0, 165, 255)  # Orange
                cand = dec.student_name or "Unknown"
                label = f"Uncertain: {cand} ({dec.similarity:.2f})"
                sub_label = dec.reason[:30] + "..." if len(dec.reason) > 30 else dec.reason
            else:  # unknown
                # Crimson / Red
                box_color = (60, 60, 230)
                label = f"Unknown ({dec.similarity:.2f})"
                sub_label = "Not in database"
                
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw semi-transparent header tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            tag_h = text_h + 10
            tag_w = max(text_w + 12, 100)
            
            tag_y1 = max(0, y1 - tag_h)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(w_img, x1 + tag_w)
            
            # Draw background rectangle for label
            cv2.rectangle(annotated_image, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
            cv2.putText(
                annotated_image,
                label,
                (tag_x1 + 6, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )
            
        return attendance_result, annotated_image, unknown_crops_info
