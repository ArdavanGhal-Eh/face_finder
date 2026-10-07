"""
Students Management Router
CRUD for student profiles, consent tracking, and reference biometric image enrollment.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel, Field

from app.config import STUDENTS_DIR, settings
from app.database import (
    create_student,
    get_all_students,
    get_student_by_id,
    update_student,
    delete_student,
    add_student_image,
    delete_student_image
)
from app.core.pipeline import FacePipeline

router = APIRouter(prefix="/api/students", tags=["Students"])


class StudentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    student_number: str = Field(..., min_length=2, max_length=50)
    has_consent: bool = Field(default=True, description="Explicit biometric authorization")
    status: str = Field(default="active")


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    student_number: Optional[str] = None
    has_consent: Optional[bool] = None
    status: Optional[str] = None


@router.get("")
def list_students(include_archived: bool = False):
    """Returns list of students with enrolled image counts."""
    return get_all_students(include_archived=include_archived)


@router.post("", status_code=status.HTTP_201_CREATED)
def add_student(student: StudentCreate):
    """Enrolls a new student identity into the database."""
    if settings.require_biometric_consent and not student.has_consent:
        raise HTTPException(
            status_code=400,
            detail="Biometric consent is required by institutional policy before creating student profile."
        )
    try:
        sid = create_student(
            student_number=student.student_number,
            name=student.name,
            has_consent=student.has_consent,
            status=student.status
        )
        return {"id": sid, "message": "Student created successfully"}
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=409, detail=f"Student number '{student.student_number}' already exists.")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{student_id}")
def get_student(student_id: int):
    """Retrieves full student details including gallery images."""
    data = get_student_by_id(student_id)
    if not data:
        raise HTTPException(status_code=404, detail="Student not found")
    return data


@router.put("/{student_id}")
def modify_student(student_id: int, payload: StudentUpdate):
    """Updates student demographic or consent status."""
    existing = get_student_by_id(student_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Student not found")
        
    ok = update_student(
        student_id=student_id,
        name=payload.name,
        student_number=payload.student_number,
        status=payload.status,
        has_consent=payload.has_consent
    )
    if not ok:
        raise HTTPException(status_code=400, detail="No fields updated")
    return {"message": "Student updated successfully"}


@router.delete("/{student_id}")
def remove_student(student_id: int):
    """
    Hard-deletes student profile, reference images, and biometric embeddings from disk and DB.
    Guarantees Right to Erasure / GDPR compliance.
    """
    existing = get_student_by_id(student_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Student not found")
        
    ok = delete_student(student_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete student")
    return {"message": "Student and all associated biometric data permanently erased"}


@router.post("/{student_id}/images")
async def upload_reference_image(
    student_id: int,
    file: UploadFile = File(...),
    is_primary: bool = Form(False)
):
    """
    Uploads a reference photo for a student, verifies face detection,
    computes 512-D ArcFace embedding, and saves to database.
    """
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
        
    if not student["has_consent"]:
        raise HTTPException(status_code=403, detail="Student has not provided biometric consent")
        
    # Read image
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
        
    # Process through pipeline
    pipeline = FacePipeline.get_instance()
    embedding, aligned_crop, quality, err = pipeline.extract_face_for_enrollment(img_bgr)
    
    if err:
        raise HTTPException(status_code=422, detail=err)
        
    # Save image file to local storage
    student_folder = STUDENTS_DIR / f"student_{student_id}"
    student_folder.mkdir(parents=True, exist_ok=True)
    
    filename = f"ref_{len(student.get('images', [])) + 1}_{file.filename}"
    save_path = student_folder / filename
    
    # Save aligned crop or original image
    save_img = aligned_crop if aligned_crop is not None else img_bgr
    cv2.imwrite(str(save_path), save_img)
    
    # Save embedding in DB
    img_id = add_student_image(
        student_id=student_id,
        file_path=str(save_path),
        embedding=embedding,
        face_quality=quality,
        is_primary=is_primary
    )
    
    return {
        "id": img_id,
        "file_path": str(save_path),
        "face_quality": round(quality, 2),
        "is_primary": is_primary,
        "message": "Reference face enrolled successfully"
    }


@router.delete("/images/{image_id}")
def remove_reference_image(image_id: int):
    """Deletes an enrolled reference image and its vector embedding."""
    ok = delete_student_image(image_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"message": "Reference image removed"}
