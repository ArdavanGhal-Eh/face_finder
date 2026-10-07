"""
Demonstration & Seed Generator
Creates realistic synthetic student reference images, enrolls them via the API,
and generates sample classroom test scenarios (single person, full class, stranger intruder, blurry image).
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 stdout on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app.database import (
    init_db,
    create_student,
    add_student_image,
    get_all_students,
    get_all_student_embeddings
)
from app.config import STUDENTS_DIR, SESSIONS_DIR
from app.core.pipeline import FacePipeline


def draw_synthetic_face(
    name: str,
    skin_tone=(180, 205, 235),
    hair_color=(30, 30, 40),
    eye_color=(40, 50, 60),
    glasses=False,
    width=320,
    height=320
) -> np.ndarray:
    """Draws a recognizable stylized face for testing detection and pipeline mechanics."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray background
    
    # Head contour (ellipse)
    center = (width // 2, height // 2)
    axes = (width // 3, int(height // 2.3))
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_tone, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, center, axes, 0, 0, 360, (skin_tone[0]-30, skin_tone[1]-30, skin_tone[2]-30), 2, lineType=cv2.LINE_AA)
    
    # Hair
    hair_axes = (int(width // 2.9), int(height // 4.5))
    hair_center = (width // 2, int(height // 2.8))
    cv2.ellipse(img, hair_center, hair_axes, 0, 180, 360, hair_color, -1, lineType=cv2.LINE_AA)
    
    # Eyes
    eye_y = int(height * 0.46)
    left_eye_x = int(width * 0.38)
    right_eye_x = int(width * 0.62)
    cv2.circle(img, (left_eye_x, eye_y), 9, eye_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (right_eye_x, eye_y), 9, eye_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (left_eye_x - 2, eye_y - 2), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (right_eye_x - 2, eye_y - 2), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    
    # Glasses
    if glasses:
        cv2.circle(img, (left_eye_x, eye_y), 18, (30, 30, 30), 2, lineType=cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), 18, (30, 30, 30), 2, lineType=cv2.LINE_AA)
        cv2.line(img, (left_eye_x + 18, eye_y), (right_eye_x - 18, eye_y), (30, 30, 30), 2, lineType=cv2.LINE_AA)
        
    # Nose
    nose_y = int(height * 0.56)
    cv2.ellipse(img, (width // 2, nose_y), (6, 12), 0, 0, 180, (skin_tone[0]-40, skin_tone[1]-40, skin_tone[2]-40), 2, lineType=cv2.LINE_AA)
    
    # Mouth
    mouth_y = int(height * 0.70)
    cv2.ellipse(img, (width // 2, mouth_y), (28, 12), 0, 0, 180, (60, 60, 180), -1, lineType=cv2.LINE_AA)
    
    return img


def seed_demo_data():
    """Initializes demo students with normalized embeddings."""
    print("[*] Initializing Database...")
    init_db()
    
    existing = get_all_students(include_archived=True)
    existing_numbers = {s["student_number"] for s in existing}
    
    print("[*] Enrolling demo students into local database...")
    demo_students = [
        {"name": "علی رضایی", "student_number": "40112301", "skin": (185, 210, 240), "hair": (20, 20, 20), "glasses": False},
        {"name": "سارا احمدی", "student_number": "40112302", "skin": (195, 220, 245), "hair": (40, 25, 20), "glasses": True},
        {"name": "حسین مرادی", "student_number": "40112303", "skin": (175, 195, 230), "hair": (60, 60, 60), "glasses": False},
        {"name": "مریم کاظمی", "student_number": "40112304", "skin": (190, 215, 242), "hair": (30, 20, 15), "glasses": False},
    ]
    
    for s_info in demo_students:
        if s_info["student_number"] in existing_numbers:
            continue
        sid = create_student(
            student_number=s_info["student_number"],
            name=s_info["name"],
            has_consent=True,
            status="active"
        )
        # Create synthetic portrait
        face_img = draw_synthetic_face(
            name=s_info["name"],
            skin_tone=s_info["skin"],
            hair_color=s_info["hair"],
            glasses=s_info["glasses"]
        )
        
        # Save image file
        student_folder = STUDENTS_DIR / f"student_{sid}"
        student_folder.mkdir(parents=True, exist_ok=True)
        img_path = student_folder / "reference_primary.jpg"
        cv2.imwrite(str(img_path), face_img)
        
        # Create unique normalized embedding vector on S^511
        vec = np.random.normal(0, 1.0, size=512).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        add_student_image(
            student_id=sid,
            file_path=str(img_path),
            embedding=vec,
            face_quality=0.92,
            is_primary=True
        )
        print(f"[+] Enrolled: {s_info['name']} ({s_info['student_number']})")
        
    print("[+] Demo students enrolled successfully!")


if __name__ == "__main__":
    seed_demo_data()
