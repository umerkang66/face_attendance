import os
from pathlib import Path

# Base directory of the project (root of the workspace)
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder paths
DATASET_DIR = BASE_DIR / "dataset"
ENCODINGS_DIR = BASE_DIR / "encodings"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"
APP_DIR = BASE_DIR / "app"

# Ensure essential directories exist
for directory in [DATASET_DIR, ENCODINGS_DIR, DATABASE_DIR, LOGS_DIR, DOCS_DIR, APP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# SQLite database file path
DB_PATH = DATABASE_DIR / "attendance.db"

# Face Recognition parameters
RECOGNITION_THRESHOLD = 0.6  # Default distance threshold for dlib face recognition (lower is stricter)

# Camera configuration
CAMERA_INDEX = 0  # Default system camera
