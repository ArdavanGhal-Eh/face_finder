"""
Full 27-Student Presence Assignment & Visualization
Face Finder - Low-Threshold Assignment Test
"""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from app.database import (
    init_db,
    get_all_student_embeddings,
    save_attendance_session
)
from app.core.pipeline import FacePipeline
from app.core.matcher import compute_similarity_matrix
from scipy.optimize import linear_sum_assignment

CLASSROOM_PHOTO_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\photo_2026-10-08_01-16-44.jpg")
DESKTOP_OUTPUT_PATH = Path(r"C:\Users\ardav\OneDrive\Desktop\classroom_attendance_result.jpg")
SESSION_OUTPUT_PATH = PROJECT_DIR / "app" / "storage" / "sessions" / "classroom_attendance_result.jpg"


def run_full_presence_visualization():
    print("[*] Running Full 27-Student Assignment on Classroom Photo...")
    pipeline = FacePipeline.get_instance()
    
    classroom_bgr = cv2.imread(str(CLASSROOM_PHOTO_PATH))
    h_img, w_img = classroom_bgr.shape[:2]
    
    gallery = get_all_student_embeddings()
    student_ids = list(gallery.keys())
    
    raw_faces = pipeline.app.get(classroom_bgr)
    print(f"[+] Detected {len(raw_faces)} faces in classroom photo.")
    
    det_embs = [f.normed_embedding for f in raw_faces]
    sim_matrix, col_student_ids = compute_similarity_matrix(det_embs, gallery)
    
    # Hungarian 1-to-1 matching
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assigned_faces = {}  # face_idx -> (student_col, similarity)
    for r, c in zip(row_ind, col_ind):
        assigned_faces[r] = (c, sim_matrix[r, c])
        
    annotated = classroom_bgr.copy()
    present_records = []
    unknown_detections = []
    
    for i, face in enumerate(raw_faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        if i in assigned_faces:
            c, sim = assigned_faces[i]
            sid = col_student_ids[c]
            student = gallery[sid]
            
            # Green Box for Present Student
            color = (36, 210, 80)  # BGR Emerald Green
            label = f"{student['name']} ({sim*100:.1f}%)"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Label tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
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
            # Red Box for Unknown Face (outside the 27 students)
            color = (50, 50, 240)  # Crimson Red
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
            
    # Save output to Desktop and Session Storage
    cv2.imwrite(str(DESKTOP_OUTPUT_PATH), annotated)
    cv2.imwrite(str(SESSION_OUTPUT_PATH), annotated)
    
    # Save to database
    save_attendance_session(
        session_name="Full 27-Student Force-Assignment Test",
        classroom_image_path=str(CLASSROOM_PHOTO_PATH),
        annotated_image_path=str(DESKTOP_OUTPUT_PATH),
        match_threshold=0.15,
        ambiguity_margin=0.01,
        present_records=present_records,
        absent_records=[],
        uncertain_records=[],
        unknown_detections=unknown_detections,
        notes="Zero-threshold best-match assignment across all 27 gallery students"
    )
    
    print("\n" + "=" * 70)
    print(f"[+] Output successfully generated!")
    print(f"    Present Students (Green Boxes): {len(present_records)} / 27")
    print(f"    Unknown Crowd Faces (Red Boxes): {len(unknown_detections)}")
    print(f"    Saved to Desktop: {DESKTOP_OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_presence_visualization()
