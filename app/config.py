"""
Configuration and Application Settings
Face Finder - Smart Classroom Attendance System
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STORAGE_DIR = APP_DIR / "storage"
STUDENTS_DIR = STORAGE_DIR / "students"
SESSIONS_DIR = STORAGE_DIR / "sessions"
DB_PATH = STORAGE_DIR / "data.db"

# Ensure storage directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class SystemSettings(BaseModel):
    """Configurable system runtime thresholds and operational parameters."""
    
    # Matching Thresholds
    match_threshold: float = Field(
        default=0.25,
        ge=0.15,
        le=0.85,
        description="Cosine similarity threshold for confirming positive identity match."
    )
    ambiguity_margin: float = Field(
        default=0.03,
        ge=0.01,
        le=0.30,
        description="Minimum gap between Top-1 and Top-2 similarity scores. Below this gap, match is labeled Uncertain."
    )
    
    # Face Quality & Detection Filtering
    min_face_size: int = Field(
        default=16,
        ge=10,
        le=150,
        description="Minimum bounding box width/height in pixels."
    )
    min_det_confidence: float = Field(
        default=0.45,
        ge=0.20,
        le=0.95,
        description="SCRFD detector confidence threshold."
    )
    min_blur_score: float = Field(
        default=25.0,
        ge=0.0,
        le=200.0,
        description="Variance of Laplacian threshold."
    )
    
    # Model & Execution
    model_name: str = Field(
        default="buffalo_l",
        description="InsightFace model pack: 'buffalo_l' (high-precision ResNet50) or 'buffalo_s' (MobileFaceNet)."
    )
    det_size_w: int = Field(default=640, description="Detection input width")
    det_size_h: int = Field(default=640, description="Detection input height")
    
    # Biometric Privacy & Compliance
    require_biometric_consent: bool = Field(
        default=True,
        description="Enforce explicit consent verification before enrolling biometric reference data."
    )
    store_annotated_sessions: bool = Field(
        default=True,
        description="Save classroom images with bounding boxes for historical auditing."
    )


# Global default instance
settings = SystemSettings()
