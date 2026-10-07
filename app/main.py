"""
FastAPI Main Application Entrypoint
Face Finder - Smart Classroom Face Recognition Attendance System
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import APP_DIR, STORAGE_DIR
from app.database import init_db, get_all_students, get_all_sessions
from app.routers import students, attendance, settings, benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown routines."""
    init_db()
    yield


app = FastAPI(
    title="Face Finder - Smart Classroom Attendance API",
    description="Privacy-focused, local on-device biometric classroom attendance verification engine.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local cross-origin development if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(students.router)
app.include_router(attendance.router)
app.include_router(settings.router)
app.include_router(benchmark.router)

# Mount Storage for serving student previews and session photos locally
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Mount Static assets
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/dashboard")
def get_dashboard_summary():
    """Returns high-level statistics for the main dashboard overview."""
    all_studs = get_all_students(include_archived=True)
    active_studs = [s for s in all_studs if s["status"] == "active"]
    sessions = get_all_sessions()
    
    last_session = sessions[0] if sessions else None
    
    return {
        "total_students": len(all_studs),
        "active_students": len(active_studs),
        "total_sessions": len(sessions),
        "last_session": last_session
    }


@app.get("/")
def serve_index():
    """Serves the single-page application dashboard frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Face Finder API is live. UI is being prepared."}
