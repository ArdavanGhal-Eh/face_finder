"""
Process Real Classroom Image with Real Student Reference Data
Face Finder - Smart Classroom Attendance Verification
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from app.database import (
    init_db,
    create_student,
    add_student_image,
    get_all_student_embeddings,
    save_attendance_session,
    get_connection
)
from app.config import STUDENTS_DIR, SESSIONS_DIR, settings
from app.core.pipeline import FacePipeline
from app.core.matcher import match_classroom_faces

STUDENTS_SOURCE_DIR = Path(r"C:\Users\ardav\OneDrive\Desktop\students")
CLASSROOM_PHOTO_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\photo_2026-10-08_01-16-44.jpg")
DESKTOP_OUTPUT_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\classroom_attendance_result.jpg")


def reset_and_enroll_students(pipeline: FacePipeline):
    """Enrolls all 27 students from Desktop/students directory."""
    print("=" * 70)
    print("[1] INITIALIZING DATABASE & ENROLLING 27 STUDENTS...")
    print("=" * 70)
    
    init_db()
    
    # Clear old data for a fresh clean test
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_images;")
        cursor.execute("DELETE FROM students;")
        cursor.execute("DELETE FROM attendance_sessions;")
        conn.commit()
        
    student_dirs = sorted([d for d in STUDENTS_SOURCE_DIR.iterdir() if d.is_dir() and d.name.startswith("student_")])
    print(f"Found {len(student_dirs)} student folders in {STUDENTS_SOURCE_DIR}.")
    
    enrolled_count = 0
    failed_enrollments = []
    
    for s_dir in student_dirs:
        folder_name = s_dir.name  # e.g. student_01
        num_str = folder_name.replace("student_", "")
        student_name = f"Student {num_str}"
        student_number = f"STU-{num_str}"
        
        # Find images in folder
        img_files = [f for f in s_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if not img_files:
            print(f"[-] No image file found in {folder_name}")
            failed_enrollments.append((folder_name, "No image file"))
            continue
            
        ref_file = img_files[0]
        ref_bgr = cv2.imread(str(ref_file))
        if ref_bgr is None:
            print(f"[-] Failed to read {ref_file.name}")
            failed_enrollments.append((folder_name, "Corrupt image"))
            continue
            
        # Detect and extract embedding
        faces = pipeline.app.get(ref_bgr)
        if not faces:
            # Try resizing if image is too small
            h, w = ref_bgr.shape[:2]
            if min(h, w) < 200:
                upscaled = cv2.resize(ref_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                faces = pipeline.app.get(upscaled)
                
        if not faces:
            print(f"[-] No face detected in {folder_name} ({ref_file.name})")
            failed_enrollments.append((folder_name, "No face detected in reference"))
            continue
            
        # Select best/largest face
        best_face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
        emb = best_face.normed_embedding
        if emb is None:
            print(f"[-] No embedding extracted for {folder_name}")
            continue
            
        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Create student in DB
        sid = create_student(
            student_number=student_number,
            name=student_name,
            has_consent=True,
            status="active"
        )
        
        # Save image copy in face_finder storage
        local_dest = STUDENTS_DIR / f"student_{sid}"
        local_dest.mkdir(parents=True, exist_ok=True)
        dest_file = local_dest / ref_file.name
        cv2.imwrite(str(dest_file), ref_bgr)
        
        add_student_image(
            student_id=sid,
            file_path=str(dest_file),
            embedding=emb,
            face_quality=float(best_face.det_score),
            is_primary=True
        )
        enrolled_count += 1
        print(f"[+] Successfully enrolled: {student_name} ({student_number}) from {ref_file.name}")
        
    print(f"\n[+] Total Enrolled: {enrolled_count} / {len(student_dirs)} students.")
    if failed_enrollments:
        print(f"[-] Failed enrollments: {failed_enrollments}")
        
    return enrolled_count


def process_classroom_attendance(pipeline: FacePipeline):
    """Processes classroom photo and annotates bounding boxes (Green=Present, Red=Unknown)."""
    print("\n" + "=" * 70)
    print(f"[2] PROCESSING CLASSROOM PHOTO: {CLASSROOM_PHOTO_PATH.name}")
    print("=" * 70)
    
    classroom_bgr = cv2.imread(str(CLASSROOM_PHOTO_PATH))
    if classroom_bgr is None:
        print(f"[!] Error: Could not read classroom photo at {CLASSROOM_PHOTO_PATH}")
        return
        
    h_img, w_img = classroom_bgr.shape[:2]
    print(f"[+] Classroom Image Resolution: {w_img}x{h_img} pixels")
    
    # 1. Load enrolled gallery
    gallery = get_all_student_embeddings()
    print(f"[+] Loaded {len(gallery)} active student galleries from database.")
    
    # 2. Detect all faces in classroom photo
    raw_faces = pipeline.app.get(classroom_bgr)
    print(f"[+] SCRFD Detector found {len(raw_faces)} faces in classroom photo.")
    
    detected_faces = []
    for i, f in enumerate(raw_faces):
        x1, y1, x2, y2 = [int(v) for v in f.bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        bw = x2 - x1
        bh = y2 - y1
        face_size = min(bw, bh)
        crop = classroom_bgr[y1:y2, x1:x2]
        blur = pipeline.calculate_blur_score(crop)
        
        emb = f.normed_embedding
        if emb is not None:
            emb = np.array(emb, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
        else:
            emb = np.zeros(512, dtype=np.float32)
            
        detected_faces.append({
            "face_index": i,
            "bbox": [x1, y1, x2, y2],
            "embedding": emb,
            "det_score": float(f.det_score),
            "blur_score": blur,
            "face_size": face_size
        })
        
    # Calibrated threshold for very small back-row faces (16-20px) in wide classroom photo
    MATCH_THRESH = 0.22
    AMBIGUITY_MARGIN = 0.02
    
    attendance_result = match_classroom_faces(
        detected_faces=detected_faces,
        gallery=gallery,
        match_threshold=MATCH_THRESH,
        ambiguity_margin=AMBIGUITY_MARGIN,
        min_face_size=10,
        min_blur_score=10.0
    )
    
    # 4. Annotate image with Green (Present) and Red (Unknown)
    annotated = classroom_bgr.copy()
    
    for dec in attendance_result.all_decisions:
        x1, y1, x2, y2 = dec.bbox
        
        if dec.status == "present":
            # Vibrant Green
            color = (36, 210, 80)
            label = f"{dec.student_name} ({dec.similarity*100:.1f}%)"
            sub_label = "Present"
        elif dec.status == "uncertain":
            # Orange/Amber for uncertain
            color = (0, 165, 255)
            cand = dec.student_name or "Uncertain"
            label = f"{cand}? ({dec.similarity*100:.1f}%)"
            sub_label = "Uncertain"
        else:
            # Crimson Red for Unknown
            color = (50, 50, 240)
            label = f"Unknown ({dec.similarity*100:.1f}%)"
            sub_label = "Not in DB"
            
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        
        # Draw label tag background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = y1
        tag_x1 = x1
        tag_x2 = min(w_img, x1 + tw + 10)
        
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (tag_x1 + 5, tag_y2 - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA
        )
        
    # Save output to Desktop and storage
    cv2.imwrite(str(DESKTOP_OUTPUT_PATH), annotated)
    
    session_out_dir = SESSIONS_DIR / "classroom_attendance_result.jpg"
    cv2.imwrite(str(session_out_dir), annotated)
    
    # Save session to DB
    session_id = save_attendance_session(
        session_name="Classroom Verification - photo_2026-10-08",
        classroom_image_path=str(CLASSROOM_PHOTO_PATH),
        annotated_image_path=str(DESKTOP_OUTPUT_PATH),
        match_threshold=MATCH_THRESH,
        ambiguity_margin=AMBIGUITY_MARGIN,
        present_records=attendance_result.present,
        absent_records=attendance_result.absent,
        uncertain_records=attendance_result.uncertain,
        unknown_detections=attendance_result.unknown,
        notes="Automated benchmark verification with real student registry"
    )
    
    # Print formatted console report
    print("\n" + "=" * 70)
    print("ATTENDANCE EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Detected Faces: {attendance_result.total_detected}")
    print(f"Present (حاضر):      {len(attendance_result.present)}")
    print(f"Absent (غایب):       {len(attendance_result.absent)}")
    print(f"Uncertain (نامطمئن): {len(attendance_result.uncertain)}")
    print(f"Unknown (ناشناس):    {len(attendance_result.unknown)}")
    print("-" * 70)
    
    print("\n[+] PRESENT STUDENTS (کادر سبز):")
    for p in attendance_result.present:
        print(f"  * {p['name']} ({p['student_number']}) - Confidence: {p['confidence']*100:.1f}%")
        
    print("\n[-] ABSENT STUDENTS (در تصویر یافت نشدند):")
    for a in attendance_result.absent:
        print(f"  * {a['name']} ({a['student_number']})")
        
    if attendance_result.uncertain:
        print("\n[?] UNCERTAIN FACES (کادر نارنجی):")
        for u in attendance_result.uncertain:
            print(f"  * Face #{u['face_index']+1}: Candidate {u['name']} - Confidence: {u['confidence']*100:.1f}% (Reason: {u['reason']})")
            
    if attendance_result.unknown:
        print("\n[!] UNKNOWN FACES (کادر قرمز):")
        for unk in attendance_result.unknown:
            print(f"  * Face #{unk['face_index']+1}: Best similarity {unk['best_similarity']*100:.1f}% - Reason: {unk['reason']}")
            
    print("\n" + "=" * 70)
    print(f"[+] Annotated image successfully saved to:")
    print(f"    --> {DESKTOP_OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    pipeline = FacePipeline.get_instance()
    reset_and_enroll_students(pipeline)
    process_classroom_attendance(pipeline)
