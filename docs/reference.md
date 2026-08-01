# Project Reference and API Documentation

This reference guide outlines the directory structure, file breakdowns, configuration options, and key symbols exposed by the system's modules.

---

## 1. Directory Structure

```text
face_attendance/
├── app/
│   └── main.py
├── database/
│   └── attendance.db
├── dataset/
├── docs/
├── encodings/
├── logs/
│   └── attendance_log.txt
├── src/
│   ├── attendance.py
│   ├── config.py
│   ├── database.py
│   ├── enroll.py
│   ├── enroll_from_folder.py
│   ├── live_recognition_test.py
│   └── recognizer.py
├── .gitignore
├── requirements.txt
└── README.md
```

![System Component Flow Diagram](images/component_flow.jpg)

---

## 2. Global Configuration Reference

Configuration settings are maintained as constants within [`src/config.py`](../src/config.py):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `BASE_DIR` | Absolute path to project root | Base directory pointer for local file routing. |
| `DATASET_DIR` | `BASE_DIR / "dataset"` | Path containing enrolled student image directories. |
| `ENCODINGS_DIR` | `BASE_DIR / "encodings"` | Reserved directory for saved embedding archives. |
| `DATABASE_DIR` | `BASE_DIR / "database"` | Path locating local SQLite database files. |
| `LOGS_DIR` | `BASE_DIR / "logs"` | Path for plain-text attendance log exports. |
| `DOCS_DIR` | `BASE_DIR / "docs"` | Path holding documentation files. |
| `APP_DIR` | `BASE_DIR / "app"` | Source directory of the Streamlit dashboard app. |
| `DB_PATH` | `DATABASE_DIR / "attendance.db"` | Full path locating the SQLite database. |
| `RECOGNITION_THRESHOLD` | `0.6` | Metric distance for recognition (lower values increase matching strictness). |
| `CAMERA_INDEX` | `0` | Camera capture hardware index used by OpenCV. |

---

## 3. Module & API Reference

### 3.1. Database Engine - [`src/database.py`](../src/database.py)
This module acts as the persistence interface layer for SQLite3.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`get_db_connection`](../src/database.py#L7) | None | `sqlite3.Connection` | Establishes a connection to the database. Sets key behaviors (foreign key enforcement, row dict formats). |
| [`init_db`](../src/database.py#L14) | None | None | Generates target `students` and `attendance` database tables if they are not present. |
| [`add_student`](../src/database.py#L44) | `name: str`, `roll_number: str`, `encoding: np.ndarray` | `bool` | Serializes the face array using Pickle and inserts the student record. |
| [`get_all_students`](../src/database.py#L75) | None | `list[dict]` | Reads and deserializes all student rows with their numpy array embeddings. |
| [`mark_attendance`](../src/database.py#L108) | `student_id: int`, `date: str`, `time: str`, `status: str` | `bool` | Records an attendance entry in the database. |
| [`get_attendance_by_date`](../src/database.py#L137) | `date: str` | `list[dict]` | Selects and outputs attendance records mapped against matching student rolls. |
| [`already_marked_today`](../src/database.py#L174) | `student_id: int`, `date: str` | `bool` | Checks if a student ID already has a log entry registered on a target date. |

### 3.2. Enrollment Controller - [`src/enroll.py`](../src/enroll.py)
This module manages active camera capture prompts to enroll student identities.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`check_roll_number_exists`](../src/enroll.py#L27) | `roll_number: str` | `bool` | Checks if a roll number is already present in the database. |
| [`simulate_enrollment`](../src/enroll.py#L32) | `name: str`, `roll_number: str`, `student_dir: Path` | `bool` | Bypasses camera capture, writing random numpy matrices and mock images to file paths. |
| [`enroll_student`](../src/enroll.py#L85) | `name: str`, `roll_number: str`, `simulate: bool` | `bool` | Walks the camera stream through 10 distinct frames, extracts embeddings, averages vectors, and saves to database. |
| [`cleanup_folder`](../src/enroll.py#L240) | `folder_path: Path` | None | Purges folders containing captured media if enrollment processes are aborted. |

### 3.3. Batch Directory Importer - [`src/enroll_from_folder.py`](../src/enroll_from_folder.py)
Imports pre-sorted student folders to populate databases in batch mode.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`check_roll_number_exists`](../src/enroll_from_folder.py#L12) | `roll_number: str` | `bool` | Validates if a student profile already exists in the database. |
| [`enroll_student_from_directory`](../src/enroll_from_folder.py#L17) | `folder_path: Path`, `simulate: bool` | `bool` | Scans image files in a subfolder, calculates/averages encodings, and registers profiles. |
| [`scan_and_enroll_dataset`](../src/enroll_from_folder.py#L112) | `simulate: bool` | None | Scans root `dataset/` directories for subfolders matching `<Name>_<RollNumber>` for enrollment processing. |

### 3.4. Recognition Matching Engine - [`src/recognizer.py`](../src/recognizer.py)
Implements algorithms that cross-reference incoming camera vectors with cached enrollment matrices.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`load_known_encodings`](../src/recognizer.py#L14) | `force_reload: bool` | `tuple[list, list]` | Pulls student profiles from databases, structures and keeps their signatures in RAM caches. |
| [`recognize_faces`](../src/recognizer.py#L45) | `frame: np.ndarray`, `threshold: float` | `list[dict]` | Pinpoints faces in frames, computes spatial coordinates, and queries caches to find profile matches. |

### 3.5. Attendance Recorder - [`src/attendance.py`](../src/attendance.py)
Drives attendance processing sessions, capturing frame streams and writing logs.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`write_log`](../src/attendance.py#L19) | `message: str` | None | Appends formatted timestamp logs to the external text log destination. |
| [`mark_absent_students`](../src/attendance.py#L30) | `date_str: str` | `int` | Identifies enrolled students who failed to log attendance during a session and writes them as "Absent". |
| [`run_attendance_session`](../src/attendance.py#L68) | `simulate: bool` | None | Operates live session loops, tracking face vectors and rendering display widgets. |
| [`run_simulated_session`](../src/attendance.py#L213) | `roll_to_student_map: dict` | None | Simulates attendance sequences via CLI text outputs, updating SQLite data files. |

### 3.6. Verification Sandbox - [`src/live_recognition_test.py`](../src/live_recognition_test.py)
Provides a lightweight sandbox tool for debugging video inputs and recognition parameters without database modifications.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`run_live_recognition`](../src/live_recognition_test.py#L12) | `simulate: bool` | None | Binds camera inputs, executing HOG detections and drawing boundary labels. |
| [`run_simulated_recognition_loop`](../src/live_recognition_test.py#L109) | `known_metadata: list` | None | Drives simple terminal loops logging mock detection events. |

### 3.7. Streamlit Control Panel Interface - [`app/main.py`](../app/main.py)
This is the central web dashboard frontend containing the enrollment layout, live attendance controls, and analytical reports.

| Function | Inputs | Outputs | Description |
| :--- | :--- | :--- | :--- |
| [`show_marked_table`](../app/main.py#L253) | None | None | Queries database records and renders presence counts and tables inside Streamlit layout views. |
