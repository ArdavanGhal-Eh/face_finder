"""
Entrypoint Runner Script
Face Finder - Smart Classroom Face Recognition Attendance System
"""

import sys
import uvicorn

if __name__ == "__main__":
    banner = r"""
    ========================================================================
     ______             ______ _           _           
    |  ____|           |  ____(_)         | |          
    | |__ __ _  ___ ___| |__   _ _ __   __| | ___ _ __ 
    |  __/ _` |/ __/ _ \  __| | | '_ \ / _` |/ _ \ '__|
    | | | (_| | (_|  __/ |    | | | | | (_| |  __/ |   
    |_|  \__,_|\___\___|_|    |_|_| |_|\__,_|\___|_|   
                                                       
     Smart Classroom Biometric Attendance System (Local & On-Device)
     Detector: SCRFD | Embedder: ArcFace (512-D) | Solver: Hungarian Matching
    ========================================================================
    Starting server at: http://127.0.0.1:8000
    Open your browser to http://127.0.0.1:8000 to access the Dashboard.
    ========================================================================
    """
    print(banner)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
