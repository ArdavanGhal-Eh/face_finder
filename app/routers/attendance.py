"""
Attendance Processing Router
Handles classroom photo uploads, face matching, record persistence, and reporting/export.
"""

import io
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
import openpyxl

from app.config import SESSIONS_DIR, settings
from app.database import (
    get_all_student_embeddings,
    save_attendance_session,
    get_all_sessions,
    get_session_detail,
    delete_session,
    get_system_settings
)
from app.core.pipeline import FacePipeline

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])


@router.post("/process")
async def process_attendance(
    file: UploadFile = File(...),
    session_name: str = Form("General Classroom Session"),
    match_threshold: Optional[float] = Form(None),
    ambiguity_margin: Optional[float] = Form(None),
    notes: Optional[str] = Form(None)
):
    """
    Evaluates attendance from a classroom photo:
    1. Detects all faces (SCRFD)
    2. Extracts 512-D ArcFace embeddings
    3. Solves optimal bipartite matching with student gallery (Hungarian Algorithm)
    4. Categorizes into Present, Absent, Uncertain, and Unknown
    5. Saves session and annotated image
    """
    # Read image
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
        
    # Load system settings
    db_settings = get_system_settings()
    th_match = match_threshold if match_threshold is not None else float(db_settings.get("match_threshold", settings.match_threshold))
    th_margin = ambiguity_margin if ambiguity_margin is not None else float(db_settings.get("ambiguity_margin", settings.ambiguity_margin))
    min_face_size = int(db_settings.get("min_face_size", settings.min_face_size))
    min_blur = float(db_settings.get("min_blur_score", settings.min_blur_score))
    
    # Load enrolled gallery from DB
    gallery = get_all_student_embeddings()
    
    # Process through pipeline
    pipeline = FacePipeline.get_instance()
    attendance_result, annotated_img, _ = pipeline.process_classroom_image(
        classroom_image_bgr=img_bgr,
        gallery=gallery,
        match_threshold=th_match,
        ambiguity_margin=th_margin,
        min_face_size=min_face_size,
        min_blur_score=min_blur
    )
    
    # Save files to disk
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"session_{timestamp_str}_raw.jpg"
    annotated_filename = f"session_{timestamp_str}_annotated.jpg"
    
    raw_path = SESSIONS_DIR / raw_filename
    annotated_path = SESSIONS_DIR / annotated_filename
    
    cv2.imwrite(str(raw_path), img_bgr)
    cv2.imwrite(str(annotated_path), annotated_img)
    
    # Save session records into DB
    session_id = save_attendance_session(
        session_name=session_name.strip(),
        classroom_image_path=str(raw_path),
        annotated_image_path=str(annotated_path),
        match_threshold=th_match,
        ambiguity_margin=th_margin,
        present_records=attendance_result.present,
        absent_records=attendance_result.absent,
        uncertain_records=attendance_result.uncertain,
        unknown_detections=attendance_result.unknown,
        notes=notes
    )
    
    # Convert annotated image to base64 or serve via static URL
    # Relative path for front-end consumption: /storage/sessions/<filename>
    annotated_url = f"/storage/sessions/{annotated_filename}"
    raw_url = f"/storage/sessions/{raw_filename}"
    
    return {
        "session_id": session_id,
        "session_name": session_name,
        "timestamp": datetime.now().isoformat(),
        "total_detected_faces": attendance_result.total_detected,
        "present_count": len(attendance_result.present),
        "absent_count": len(attendance_result.absent),
        "uncertain_count": len(attendance_result.uncertain),
        "unknown_count": len(attendance_result.unknown),
        "thresholds_used": {
            "match_threshold": th_match,
            "ambiguity_margin": th_margin
        },
        "annotated_image_url": annotated_url,
        "raw_image_url": raw_url,
        "present": attendance_result.present,
        "absent": attendance_result.absent,
        "uncertain": attendance_result.uncertain,
        "unknown": attendance_result.unknown
    }


@router.get("/sessions")
def list_sessions():
    """Returns list of past attendance sessions."""
    sessions = get_all_sessions()
    # Normalize image paths to web URLs
    for s in sessions:
        if s["annotated_image_path"]:
            p = Path(s["annotated_image_path"])
            s["annotated_image_url"] = f"/storage/sessions/{p.name}"
        if s["classroom_image_path"]:
            p = Path(s["classroom_image_path"])
            s["classroom_image_url"] = f"/storage/sessions/{p.name}"
    return sessions


@router.get("/sessions/{session_id}")
def get_session(session_id: int):
    """Retrieves full details of a specific session."""
    session = get_session_detail(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session["annotated_image_path"]:
        p = Path(session["annotated_image_path"])
        session["annotated_image_url"] = f"/storage/sessions/{p.name}"
    if session["classroom_image_path"]:
        p = Path(session["classroom_image_path"])
        session["classroom_image_url"] = f"/storage/sessions/{p.name}"
        
    return session


@router.delete("/sessions/{session_id}")
def delete_attendance_session(session_id: int):
    """Deletes an attendance session and its saved photos."""
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Attendance session deleted"}


@router.get("/sessions/{session_id}/export")
def export_session_report(session_id: int, format: str = "csv"):
    """Exports attendance report as CSV or Excel."""
    session = get_session_detail(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session_title = session["session_name"]
    date_str = session["timestamp"][:10]
    
    if format.lower() == "excel":
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Attendance Report"
        
        # Header info
        ws.append(["Smart Classroom Face Finder - Attendance Report"])
        ws.append(["Session Name:", session_title])
        ws.append(["Date & Time:", session["timestamp"]])
        ws.append(["Total Detected Faces:", session["total_detected_faces"]])
        ws.append(["Present:", session["present_count"], "Absent:", session["absent_count"], "Uncertain:", session["uncertain_count"], "Unknown:", session["unknown_count"]])
        ws.append([])
        
        # Student Attendance Table
        ws.append(["Category", "Student Name", "Student Number", "Confidence", "Reason"])
        for rec in session["records"]:
            ws.append([
                rec["status"].upper(),
                rec["student_name"],
                rec["student_number"],
                f"{rec['confidence']:.2f}" if rec['confidence'] > 0 else "-",
                rec["reason"] or ""
            ])
            
        # Unknown detections
        if session["unknowns"]:
            ws.append([])
            ws.append(["Unknown Detections"])
            ws.append(["Face #", "Best Similarity", "Closest Candidate", "Reason"])
            for unk in session["unknowns"]:
                ws.append([
                    f"Face #{unk['face_index'] + 1}",
                    f"{unk['best_similarity']:.2f}",
                    unk["best_candidate_name"] or "None",
                    unk["reason"]
                ])
                
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        filename = f"attendance_{session_id}_{date_str}.xlsx"
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        # Default CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Category", "Student Name", "Student Number", "Confidence", "Reason"])
        
        for rec in session["records"]:
            writer.writerow([
                rec["status"].upper(),
                rec["student_name"],
                rec["student_number"],
                f"{rec['confidence']:.2f}" if rec['confidence'] > 0 else "-",
                rec["reason"] or ""
            ])
            
        for unk in session["unknowns"]:
            writer.writerow([
                "UNKNOWN",
                f"Face #{unk['face_index'] + 1}",
                "-",
                f"{unk['best_similarity']:.2f}",
                unk["reason"]
            ])
            
        filename = f"attendance_{session_id}_{date_str}.csv"
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
