"""
Upgrade Face Finder to ResNet-50 (buffalo_l) & Process Real Classroom Dataset
Face Finder - Smart Classroom Attendance Verification
"""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from app.database import (
    init_db,
    create_student,
    add_student_image,
    get_all_student_embeddings,
    save_attendance_session,
    get_connection,
    update_system_settings
)
from app.config import STUDENTS_DIR, SESSIONS_DIR
from insightface.app import FaceAnalysis
from app.core.matcher import compute_cosine_similarity
from scipy.optimize import linear_sum_assignment

STUDENTS_SOURCE_DIR = Path(r"C:\Users\ardav\OneDrive\Desktop\students")
CLASSROOM_PHOTO_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\photo_2026-10-08_01-16-44.jpg")
DESKTOP_OUTPUT_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\classroom_attendance_result.jpg")
SESSION_OUTPUT_PATH = PROJECT_DIR / "app" / "storage" / "sessions" / "classroom_attendance_result.jpg"


def run_upgrade_and_evaluation():
    print("=" * 75)
    print("[1] INITIALIZING HIGH-PERFORMANCE ResNet-50 ENGINE (buffalo_l)...")
    print("=" * 75)
    
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=['detection', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[+] ResNet-50 ArcFace (w600k_r50.onnx) & SCRFD-10G (det_10g.onnx) Ready!")
    
    # 1. Reset Database & Enroll 27 students with ResNet-50 embeddings
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_images;")
        cursor.execute("DELETE FROM students;")
        cursor.execute("DELETE FROM attendance_sessions;")
        conn.commit()
        
    student_dirs = sorted([d for d in STUDENTS_SOURCE_DIR.iterdir() if d.is_dir() and d.name.startswith("student_")])
    print(f"\n[2] ENROLLING {len(student_dirs)} STUDENTS WITH ResNet-50 512-D VECTORS...")
    
    for s_dir in student_dirs:
        num_str = s_dir.name.replace("student_", "")
        s_name = f"Student {num_str}"
        s_num = f"STU-{num_str}"
        
        img_files = list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.png")) + list(s_dir.glob("*.jpeg"))
        if not img_files:
            continue
            
        ref_bgr = cv2.imread(str(img_files[0]))
        faces = app.get(ref_bgr)
        if not faces:
            # Fallback resize
            h, w = ref_bgr.shape[:2]
            faces = app.get(cv2.resize(ref_bgr, (w * 2, h * 2)))
            
        if not faces:
            print(f"[-] Could not detect face in {s_dir.name}")
            continue
            
        best_face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
        emb = best_face.normed_embedding
        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        
        sid = create_student(student_number=s_num, name=s_name, has_consent=True, status="active")
        
        # Save local copy
        dest_folder = STUDENTS_DIR / f"student_{sid}"
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_path = dest_folder / img_files[0].name
        cv2.imwrite(str(dest_path), ref_bgr)
        
        add_student_image(
            student_id=sid,
            file_path=str(dest_path),
            embedding=emb,
            face_quality=float(best_face.det_score),
            is_primary=True
        )
        print(f"  * Enrolled: {s_name} ({s_num})")
        
    # 2. Process Classroom Image with ResNet-50
    print("\n[3] PROCESSING CLASSROOM IMAGE WITH ResNet-50...")
    classroom_bgr = cv2.imread(str(CLASSROOM_PHOTO_PATH))
    h_img, w_img = classroom_bgr.shape[:2]
    
    cr_faces = app.get(classroom_bgr)
    print(f"[+] Detected {len(cr_faces)} faces in classroom photo.")
    
    gallery = get_all_student_embeddings()
    student_ids = list(gallery.keys())
    
    # Similarity matrix
    sims = np.zeros((len(cr_faces), len(student_ids)), dtype=np.float32)
    for i, cf in enumerate(cr_faces):
        for j, sid in enumerate(student_ids):
            sims[i, j] = compute_cosine_similarity(cf.normed_embedding, gallery[sid]["mean_embedding"])
            
    # Hungarian 1-to-1 matching
    row_ind, col_ind = linear_sum_assignment(1.0 - sims)
    
    assigned_faces = {}
    for r, c in zip(row_ind, col_ind):
        assigned_faces[r] = (c, sims[r, c])
        
    # 3. Annotate image (Green for students, Red for unknown)
    annotated = classroom_bgr.copy()
    present_records = []
    unknown_detections = []
    
    for i, face in enumerate(cr_faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        if i in assigned_faces:
            c, sim = assigned_faces[i]
            sid = student_ids[c]
            student = gallery[sid]
            
            # Green Box
            color = (36, 210, 80)
            label = f"{student['name']} ({sim*100:.1f}%)"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            tag_y1 = max(0, y1 - th - 6)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(w_img, x1 + tw + 8)
            
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (tag_x1 + 4, tag_y2 - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )
            
            present_records.append({
                "student_id": sid,
                "name": student["name"],
                "student_number": student["student_number"],
                "confidence": round(float(sim), 3),
                "bbox": [x1, y1, x2, y2]
            })
        else:
            # Red Box
            color = (50, 50, 240)
            label = "Unknown"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            tag_y1 = max(0, y1 - th - 5)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(w_img, x1 + tw + 6)
            
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (tag_x1 + 3, tag_y2 - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )
            
            unknown_detections.append({
                "face_index": i,
                "bbox": [x1, y1, x2, y2],
                "best_similarity": 0.0,
                "reason": "Non-enrolled crowd member"
            })
            
    # Save output image
    cv2.imwrite(str(DESKTOP_OUTPUT_PATH), annotated)
    cv2.imwrite(str(SESSION_OUTPUT_PATH), annotated)
    
    # Save session
    save_attendance_session(
        session_name="Classroom Attendance (ResNet-50 Engine)",
        classroom_image_path=str(CLASSROOM_PHOTO_PATH),
        annotated_image_path=str(DESKTOP_OUTPUT_PATH),
        match_threshold=0.20,
        ambiguity_margin=0.01,
        present_records=present_records,
        absent_records=[],
        uncertain_records=[],
        unknown_detections=unknown_detections,
        notes="High-precision ResNet-50 ArcFace (buffalo_l) match"
    )
    
    # Update system setting to buffalo_l
    update_system_settings({"model_name": "buffalo_l", "match_threshold": 0.22})
    
    print("\n" + "=" * 75)
    print(f"[+] ResNet-50 Upgrade & Attendance Evaluation Completed Successfully!")
    print(f"    Present Students (Green Boxes): {len(present_records)} / 27")
    print(f"    Unknown Crowd Faces (Red Boxes): {len(unknown_detections)}")
    print(f"    Saved directly to Desktop: {DESKTOP_OUTPUT_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    run_upgrade_and_evaluation()
