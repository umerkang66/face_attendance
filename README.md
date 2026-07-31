# Face Recognition Attendance Management System

This is a computer vision course project that implements an automated student attendance tracking system using facial recognition.

## Tech Stack

- **Language:** Python 3.x
- **Computer Vision:** OpenCV, `face_recognition` (dlib-based embeddings)
- **Database:** SQLite3 (standard library)
- **UI/Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Image Processing:** Pillow (PIL)

## Folder Structure

```text
face_attendance/
├── app/                  # Streamlit UI files
│   └── app.py            # Main Streamlit dashboard application (placeholder)
├── database/             # Stores SQLite database files
│   └── attendance.db     # Automatically created SQLite database
├── dataset/              # Enrolled student face images
├── docs/                 # Project documentation and reports
├── encodings/            # Stored face embeddings (e.g., pickle/numpy files)
├── logs/                 # Log files for runtime tracking
├── src/                  # Core modules
│   ├── config.py         # Global configuration and constants
│   └── database.py       # SQLite database initialization and queries
├── requirements.txt      # Project library dependencies
└── README.md             # Project documentation (this file)
```

## Database Schema

The database consists of two tables:

1. **`students`**: Stores student profile and enrolled 128D face embedding BLOBs.
   - `id` (INTEGER, Primary Key, Autoincrement)
   - `name` (TEXT, Not Null)
   - `roll_number` (TEXT, Unique, Not Null)
   - `encoding` (BLOB, serialized NumPy face embedding array)
   - `date_enrolled` (TIMESTAMP, default CURRENT_TIMESTAMP)
2. **`attendance`**: Tracks daily attendance status logs.
   - `id` (INTEGER, Primary Key, Autoincrement)
   - `student_id` (INTEGER, Foreign Key referencing `students(id)`)
   - `date` (TEXT, Not Null)
   - `time` (TEXT, Not Null)
   - `status` (TEXT, Not Null)

## Setup and Scaffolding

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize database structure:
   ```bash
   python -m src.database
   ```
