"""
SQLite Database Layer with Vector Storage & Relational Models
Face Finder - Smart Classroom Attendance System
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from app.config import DB_PATH, STUDENTS_DIR, SESSIONS_DIR, settings


def get_connection() -> sqlite3.Connection:
    """Creates a sqlite3 connection with foreign key support and dict-like rows."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables and default configuration settings."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Students Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            has_consent INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        
        # 2. Student Reference Images & Embeddings Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            face_quality REAL DEFAULT 1.0,
            is_primary INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """)
        
        # 3. Attendance Sessions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            classroom_image_path TEXT,
            annotated_image_path TEXT,
            total_detected_faces INTEGER DEFAULT 0,
            present_count INTEGER DEFAULT 0,
            absent_count INTEGER DEFAULT 0,
            uncertain_count INTEGER DEFAULT 0,
            unknown_count INTEGER DEFAULT 0,
            match_threshold_used REAL NOT NULL,
            ambiguity_margin_used REAL NOT NULL,
            notes TEXT
        );
        """)
        
        # 4. Attendance Records Table (Per Student Per Session)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT NOT NULL, -- 'present', 'absent', 'uncertain'
            confidence REAL DEFAULT 0.0,
            bbox_json TEXT,
            reason TEXT,
            FOREIGN KEY(session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """)
        
        # 5. Unknown Detections Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS unknown_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            bbox_json TEXT NOT NULL,
            best_similarity REAL DEFAULT 0.0,
            best_candidate_name TEXT,
            reason TEXT NOT NULL,
            crop_path TEXT,
            FOREIGN KEY(session_id) REFERENCES attendance_sessions(id) ON DELETE CASCADE
        );
        """)
        
        # 6. Persistent System Settings Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        
        # Seed default settings if not exists
        for key, val in settings.model_dump().items():
            cursor.execute(
                "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?);",
                (key, json.dumps(val))
            )
        
        conn.commit()


# =========================================================================
# Vector Serialization Utilities (512-D float32 ArcFace embeddings)
# =========================================================================

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serializes a float32 NumPy embedding array to raw bytes for BLOB storage."""
    emb = np.ascontiguousarray(embedding, dtype=np.float32)
    return emb.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserializes raw bytes back to a normalized float32 NumPy embedding array."""
    emb = np.frombuffer(blob, dtype=np.float32)
    # Ensure L2 normalized
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb


# =========================================================================
# Student Database Queries
# =========================================================================

def create_student(
    student_number: str,
    name: str,
    has_consent: bool = True,
    status: str = "active"
) -> int:
    """Creates a new student record and returns the created student_id."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (student_number, name, status, has_consent, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (student_number.strip(), name.strip(), status, 1 if has_consent else 0, now, now)
        )
        conn.commit()
        return cursor.lastrowid


def get_all_students(include_archived: bool = False) -> List[Dict[str, Any]]:
    """Retrieves all students with image counts and primary preview image."""
    query = """
    SELECT s.id, s.student_number, s.name, s.status, s.has_consent, s.created_at, s.updated_at,
           COUNT(img.id) as image_count,
           (SELECT file_path FROM student_images WHERE student_id = s.id ORDER BY is_primary DESC, id ASC LIMIT 1) as preview_image
    FROM students s
    LEFT JOIN student_images img ON s.id = img.student_id
    """
    if not include_archived:
        query += " WHERE s.status = 'active'"
    query += " GROUP BY s.id ORDER BY s.name ASC;"
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single student by id along with their enrolled images."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE id = ?;", (student_id,))
        student = cursor.fetchone()
        if not student:
            return None
        
        student_dict = dict(student)
        cursor.execute(
            "SELECT id, file_path, face_quality, is_primary, created_at FROM student_images WHERE student_id = ? ORDER BY is_primary DESC, id ASC;",
            (student_id,)
        )
        student_dict["images"] = [dict(img) for img in cursor.fetchall()]
        return student_dict


def update_student(
    student_id: int,
    name: Optional[str] = None,
    student_number: Optional[str] = None,
    status: Optional[str] = None,
    has_consent: Optional[bool] = None
) -> bool:
    """Updates student demographic/consent metadata."""
    now = datetime.now().isoformat()
    fields = []
    params = []
    
    if name is not None:
        fields.append("name = ?")
        params.append(name.strip())
    if student_number is not None:
        fields.append("student_number = ?")
        params.append(student_number.strip())
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if has_consent is not None:
        fields.append("has_consent = ?")
        params.append(1 if has_consent else 0)
        
    if not fields:
        return False
        
    fields.append("updated_at = ?")
    params.append(now)
    params.append(student_id)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE students SET {', '.join(fields)} WHERE id = ?;", params)
        conn.commit()
        return cursor.rowcount > 0


def delete_student(student_id: int) -> bool:
    """Hard-deletes student and all associated biometric images and embeddings (Privacy Compliance)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Find images to delete from disk
        cursor.execute("SELECT file_path FROM student_images WHERE student_id = ?;", (student_id,))
        images = cursor.fetchall()
        for img in images:
            p = Path(img["file_path"])
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        
        cursor.execute("DELETE FROM students WHERE id = ?;", (student_id,))
        conn.commit()
        return cursor.rowcount > 0


def add_student_image(
    student_id: int,
    file_path: str,
    embedding: np.ndarray,
    face_quality: float = 1.0,
    is_primary: bool = False
) -> int:
    """Enrolls a reference face image and its 512-D ArcFace embedding."""
    now = datetime.now().isoformat()
    blob = serialize_embedding(embedding)
    with get_connection() as conn:
        cursor = conn.cursor()
        if is_primary:
            cursor.execute("UPDATE student_images SET is_primary = 0 WHERE student_id = ?;", (student_id,))
        
        cursor.execute(
            """
            INSERT INTO student_images (student_id, file_path, embedding, face_quality, is_primary, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (student_id, file_path, blob, face_quality, 1 if is_primary else 0, now)
        )
        conn.commit()
        return cursor.lastrowid


def delete_student_image(image_id: int) -> bool:
    """Deletes a single reference image from disk and DB."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM student_images WHERE id = ?;", (image_id,))
        row = cursor.fetchone()
        if row:
            p = Path(row["file_path"])
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
            cursor.execute("DELETE FROM student_images WHERE id = ?;", (image_id,))
            conn.commit()
            return True
        return False


def get_all_student_embeddings() -> Dict[int, Dict[str, Any]]:
    """
    Loads all active student identities and their embeddings into memory.
    Returns:
        dict: {
            student_id: {
                "name": str,
                "student_number": str,
                "embeddings": List[np.ndarray],  # individual exemplar embeddings
                "mean_embedding": np.ndarray      # normalized centroid embedding
            }
        }
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.name, s.student_number, img.embedding, img.face_quality
            FROM students s
            JOIN student_images img ON s.id = img.student_id
            WHERE s.status = 'active' AND s.has_consent = 1
            ORDER BY s.id ASC, img.is_primary DESC;
        """)
        rows = cursor.fetchall()
        
    gallery: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        sid = r["id"]
        emb = deserialize_embedding(r["embedding"])
        if sid not in gallery:
            gallery[sid] = {
                "id": sid,
                "name": r["name"],
                "student_number": r["student_number"],
                "embeddings": [],
                "mean_embedding": None
            }
        gallery[sid]["embeddings"].append(emb)
        
    # Compute normalized mean vector for each student
    for sid, data in gallery.items():
        if data["embeddings"]:
            stacked = np.stack(data["embeddings"], axis=0)
            mean_vec = np.mean(stacked, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-6:
                mean_vec = mean_vec / norm
            data["mean_embedding"] = mean_vec
            
    return gallery


# =========================================================================
# Attendance Sessions & Records
# =========================================================================

def save_attendance_session(
    session_name: str,
    classroom_image_path: str,
    annotated_image_path: str,
    match_threshold: float,
    ambiguity_margin: float,
    present_records: List[Dict[str, Any]],
    absent_records: List[Dict[str, Any]],
    uncertain_records: List[Dict[str, Any]],
    unknown_detections: List[Dict[str, Any]],
    notes: Optional[str] = None
) -> int:
    """Stores full classroom attendance evaluation session and detailed records."""
    now = datetime.now().isoformat()
    total_detected = len(present_records) + len(uncertain_records) + len(unknown_detections)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance_sessions (
                session_name, timestamp, classroom_image_path, annotated_image_path,
                total_detected_faces, present_count, absent_count, uncertain_count,
                unknown_count, match_threshold_used, ambiguity_margin_used, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                session_name, now, classroom_image_path, annotated_image_path,
                total_detected, len(present_records), len(absent_records),
                len(uncertain_records), len(unknown_detections),
                match_threshold, ambiguity_margin, notes
            )
        )
        session_id = cursor.lastrowid
        
        # Save Present records
        for rec in present_records:
            cursor.execute(
                """
                INSERT INTO attendance_records (session_id, student_id, status, confidence, bbox_json, reason)
                VALUES (?, ?, 'present', ?, ?, ?);
                """,
                (session_id, rec["student_id"], rec.get("confidence", 0.0), json.dumps(rec.get("bbox", [])), rec.get("reason", "Confident Match"))
            )
            
        # Save Absent records
        for rec in absent_records:
            cursor.execute(
                """
                INSERT INTO attendance_records (session_id, student_id, status, confidence, bbox_json, reason)
                VALUES (?, ?, 'absent', 0.0, NULL, ?);
                """,
                (session_id, rec["student_id"], "Not detected in classroom image")
            )
            
        # Save Uncertain records
        for rec in uncertain_records:
            cursor.execute(
                """
                INSERT INTO attendance_records (session_id, student_id, status, confidence, bbox_json, reason)
                VALUES (?, ?, 'uncertain', ?, ?, ?);
                """,
                (session_id, rec["student_id"], rec.get("confidence", 0.0), json.dumps(rec.get("bbox", [])), rec.get("reason", "Low confidence / ambiguity"))
            )
            
        # Save Unknown detections
        for unk in unknown_detections:
            cursor.execute(
                """
                INSERT INTO unknown_detections (session_id, face_index, bbox_json, best_similarity, best_candidate_name, reason, crop_path)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    session_id,
                    unk.get("face_index", 0),
                    json.dumps(unk.get("bbox", [])),
                    unk.get("best_similarity", 0.0),
                    unk.get("best_candidate_name", ""),
                    unk.get("reason", "No matching student identity"),
                    unk.get("crop_path", "")
                )
            )
            
        conn.commit()
        return session_id


def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieves summary list of all recorded attendance sessions."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_name, timestamp, classroom_image_path, annotated_image_path,
                   total_detected_faces, present_count, absent_count, uncertain_count,
                   unknown_count, match_threshold_used, notes
            FROM attendance_sessions
            ORDER BY timestamp DESC;
        """)
        return [dict(r) for r in cursor.fetchall()]


def get_session_detail(session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves full details of an attendance session including student records and unknown crops."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance_sessions WHERE id = ?;", (session_id,))
        session = cursor.fetchone()
        if not session:
            return None
        
        detail = dict(session)
        
        # Load attendance records
        cursor.execute("""
            SELECT ar.id, ar.student_id, ar.status, ar.confidence, ar.bbox_json, ar.reason,
                   s.name as student_name, s.student_number
            FROM attendance_records ar
            JOIN students s ON ar.student_id = s.id
            WHERE ar.session_id = ?
            ORDER BY ar.status ASC, s.name ASC;
        """, (session_id,))
        detail["records"] = [dict(r) for r in cursor.fetchall()]
        
        # Load unknown detections
        cursor.execute("""
            SELECT id, face_index, bbox_json, best_similarity, best_candidate_name, reason, crop_path
            FROM unknown_detections
            WHERE session_id = ?
            ORDER BY face_index ASC;
        """, (session_id,))
        detail["unknowns"] = [dict(u) for u in cursor.fetchall()]
        
        return detail


def delete_session(session_id: int) -> bool:
    """Deletes an attendance session and its associated files."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT classroom_image_path, annotated_image_path FROM attendance_sessions WHERE id = ?;", (session_id,))
        row = cursor.fetchone()
        if row:
            for p_str in [row["classroom_image_path"], row["annotated_image_path"]]:
                if p_str:
                    p = Path(p_str)
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            
            cursor.execute("SELECT crop_path FROM unknown_detections WHERE session_id = ?;", (session_id,))
            for unk in cursor.fetchall():
                if unk["crop_path"]:
                    p = Path(unk["crop_path"])
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                            
            cursor.execute("DELETE FROM attendance_sessions WHERE id = ?;", (session_id,))
            conn.commit()
            return True
        return False


# =========================================================================
# Settings Get / Set Persistence
# =========================================================================

def get_system_settings() -> Dict[str, Any]:
    """Retrieves all persisted settings from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM system_settings;")
        rows = cursor.fetchall()
        result = {}
        for r in rows:
            try:
                result[r["key"]] = json.loads(r["value"])
            except Exception:
                result[r["key"]] = r["value"]
        return result


def update_system_settings(new_settings: Dict[str, Any]):
    """Saves updated settings values to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        for k, v in new_settings.items():
            cursor.execute(
                "INSERT INTO system_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (k, json.dumps(v))
            )
        conn.commit()
