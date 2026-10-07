"""
Settings & System Calibration Router
Manages matching thresholds, quality filters, and hardware acceleration metadata.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import onnxruntime

from app.config import settings
from app.database import get_system_settings, update_system_settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdatePayload(BaseModel):
    match_threshold: Optional[float] = Field(None, ge=0.25, le=0.85)
    ambiguity_margin: Optional[float] = Field(None, ge=0.01, le=0.30)
    min_face_size: Optional[int] = Field(None, ge=16, le=150)
    min_det_confidence: Optional[float] = Field(None, ge=0.30, le=0.95)
    min_blur_score: Optional[float] = Field(None, ge=0.0, le=200.0)
    model_name: Optional[str] = Field(None)
    require_biometric_consent: Optional[bool] = Field(None)
    store_annotated_sessions: Optional[bool] = Field(None)


@router.get("")
def read_settings():
    """Returns current active settings, hardware acceleration status, and calibration guidelines."""
    current = get_system_settings()
    available_providers = onnxruntime.get_available_providers()
    gpu_active = any("CUDA" in p or "DirectML" in p or "Tensorrt" in p for p in available_providers)
    
    return {
        "settings": current,
        "hardware": {
            "onnx_version": onnxruntime.__version__,
            "available_providers": available_providers,
            "gpu_acceleration_available": gpu_active,
            "recommended_device": "GPU (CUDA / DirectML)" if gpu_active else "CPU (Optimized SIMD / AVX2)"
        },
        "calibration_guidelines": {
            "match_threshold": {
                "default": 0.50,
                "current": current.get("match_threshold", 0.50),
                "lower_effect": "Increases Recall (fewer Absents), but raises False Positive Risk (wrong identification).",
                "higher_effect": "Increases Precision (zero false positives), but raises False Negative Risk (more missed students)."
            },
            "ambiguity_margin": {
                "default": 0.08,
                "current": current.get("ambiguity_margin", 0.08),
                "explanation": "If similarity difference between Top-1 and Top-2 is less than this margin, the face is routed to Uncertain to prevent false attendance."
            },
            "min_blur_score": {
                "default": 45.0,
                "current": current.get("min_blur_score", 45.0),
                "explanation": "Variance of Laplacian threshold. Higher values strictly reject blurry/motion-blurred faces."
            }
        }
    }


@router.put("")
def modify_settings(payload: SettingsUpdatePayload):
    """Updates system thresholds in persistent database."""
    updates = {}
    for k, v in payload.model_dump().items():
        if v is not None:
            updates[k] = v
            
    if not updates:
        raise HTTPException(status_code=400, detail="No parameters provided for update")
        
    update_system_settings(updates)
    
    # Update in-memory settings
    for k, v in updates.items():
        if hasattr(settings, k):
            setattr(settings, k, v)
            
    return {
        "message": "Settings updated successfully",
        "updated": updates
    }
