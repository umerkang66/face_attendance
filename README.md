# Face Recognition Attendance Management System Documentation

## 1. Project Overview

The Face Recognition Attendance Management System is a computer vision-based application designed to automate student attendance tracking. The system captures face images, computes 128-dimensional embeddings using dlib-based facial signatures, manages student records in an SQLite database, and offers a Streamlit control panel interface alongside standalone command-line scripts. It provides a contact-free, automated, and secure biometric solution to replace manual roll calls and hardware tokens.

### Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| Programming Language | Python 3.8 | Core system development and script execution |
| Web Interface | Streamlit | Web-based control panel, visual dashboards, and student enrollment UI |
| Computer Vision | OpenCV (`opencv-python`) | Image capture, frame preprocessing, bounding box drawing, and camera interface |
| Facial Recognition | `dlib` / `face_recognition` | Face localization (HOG model) and 128-dimensional facial feature embedding extraction |
| Database | SQLite3 | Local storage of student profiles, serialized embeddings, and attendance records |
| Data Processing | Pandas, NumPy | Structured data manipulation, vector math, and export functionality |
| Image Manipulation | Pillow | Draw placeholder templates and manipulate images in simulation mode |

---

## 2. Architecture / How It Works

### System Flow
The application processes inputs, performs core logic, and yields outputs through the following stages:

1. Input: Video frames are captured in real-time from the local webcam via OpenCV, or raw image files are loaded from structured directories in batch mode.
2. Processing:
   - Convert frames from BGR format to RGB format.
   - Run face localization using the Histogram of Oriented Gradients (HOG) algorithm.
   - Extract a 128-dimensional floating-point vector (face embedding) for each localized face.
   - Query the cached database embeddings to calculate the Euclidean distance between the captured face and enrolled student signatures.
   - Verify matches using a distance threshold, where a distance less than or equal to the threshold is registered as a positive match.
   - Apply a 3-second debounce window to prevent multiple database writes for the same student during a single session.
3. Output: Render color-coded bounding boxes and student details on the video frames (green for verified students, red for unrecognized faces). Insert attendance logs into the database and write to the text-based log file. Mark missing students as absent at the end of the session.

### Key Design Decisions
- SQLite3 with Pickle Serialization: To avoid heavy database servers, the system uses SQLite3. Face embeddings (128D NumPy arrays) are serialized via python `pickle` into BLOB fields, allowing quick reads and writes with minimal dependencies.
- HOG Face Detection Model: The Histogram of Oriented Gradients (HOG) model is selected for real-time face detection. HOG executes efficiently on standard CPU architectures without requiring GPU hardware acceleration.
- Multi-Pose Averaging during Enrollment: The registration module captures 10 distinct facial poses (frontal, left/right profile, head tilts, smile) and calculates the mathematical average of the extracted vectors. This process stabilizes face matching by reducing the impact of expression changes and lighting variations.
- Debounce and Confirmation Management: A 3-second debounce check limits redundant database writes. A 2-second visual confirmation overlay displays the "Marked" status to confirm registration.
- Standalone Simulation Mode: Standalone command-line tools and Streamlit scripts support a `--simulate` parameter. This allows database, logic, and UI checks to run on headless systems or platforms lacking camera peripherals.

---

## 3. Project Structure

### File Tree

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

### Component Structure and Purpose

| File/Folder | Purpose | Key Contents |
| :--- | :--- | :--- |
| [`app/`](file:///D:/Workspace/datascience-projects/face_attendance/app) | Streamlit application directory | Contains web interface files |
| [`app/main.py`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py) | Web application script | Streamlit dashboards, enrollment forms, and attendance sheets |
| [`database/`](file:///D:/Workspace/datascience-projects/face_attendance/database) | Contains the sqlite database | Directory holding local database instances |
| [`database/attendance.db`](file:///D:/Workspace/datascience-projects/face_attendance/database/attendance.db) | SQLite database file | Contains the `students` and `attendance` tables |
| [`dataset/`](file:///D:/Workspace/datascience-projects/face_attendance/dataset) | Enrolled student images | Subfolders matching the format `<Name>_<RollNumber>` containing raw photos |
| [`docs/`](file:///D:/Workspace/datascience-projects/face_attendance/docs) | Project documentation storage | Markdown notes and report details |
| [`encodings/`](file:///D:/Workspace/datascience-projects/face_attendance/encodings) | Folder for serialized embeddings | Reserved for exported feature arrays |
| [`logs/`](file:///D:/Workspace/datascience-projects/face_attendance/logs) | System log directory | Folder containing log output files |
| [`logs/attendance_log.txt`](file:///D:/Workspace/datascience-projects/face_attendance/logs/attendance_log.txt) | Plaintext execution log | Text entries tracking attendance markings with timestamps |
| [`src/`](file:///D:/Workspace/datascience-projects/face_attendance/src) | Core python modules | Source files managing configurations, databases, and computer vision operations |
| [`src/attendance.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py) | Attendance session driver | Real-time capturing loops, database logger, and simulated attendance checks |
| [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py) | Global configuration settings | Path definitions, recognition thresholds, and camera hardware indices |
| [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py) | Database interface module | Table schema initializations, serialization queries, and data extraction |
| [`src/enroll.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py) | Student webcam enrollment script | Guided photo collection (10 steps), image saving, and vector averaging |
| [`src/enroll_from_folder.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py) | Bulk folder importer | Batch scanner parsing naming formats and extracting face signatures |
| [`src/live_recognition_test.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py) | Real-time recognition validator | Standalone webcam test tool displaying identification status |
| [`src/recognizer.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py) | Face matching module | Embedding caching routines and Euclidean distance matches |
| [`.gitignore`](file:///D:/Workspace/datascience-projects/face_attendance/.gitignore) | Git exclusion rules | Configuration patterns to exclude local build artifacts and datasets from commits |
| [`requirements.txt`](file:///D:/Workspace/datascience-projects/face_attendance/requirements.txt) | Python dependencies list | Installation manifest detailing libraries and compiler workarounds |

---

## 4. File-by-File Breakdown

### 4.1. [src/config.py](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py)
This configuration module establishes standard directories, sets the dlib recognition threshold, and defines camera parameters. It verifies and builds all required project directories on startup.
- Imports and dependencies: `os`, `pathlib.Path`
- Code symbols exposed: `BASE_DIR`, `DATASET_DIR`, `ENCODINGS_DIR`, `DATABASE_DIR`, `LOGS_DIR`, `DOCS_DIR`, `APP_DIR`, `DB_PATH`, `RECOGNITION_THRESHOLD`, `CAMERA_INDEX`.
- Connection to other files: Loaded by every module in the project to resolve paths and threshold boundaries.

#### Functions and Methods
Not applicable: This module exposes configuration variables and directories. It does not define functional routines or object classes.

### 4.2. [src/database.py](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py)
This database module controls interactions with SQLite. It builds the system tables, inserts student profiles, registers attendance, and processes data retrievals.
- Imports and dependencies: `sqlite3`, `pickle`, `numpy` (as `np`), `datetime.datetime`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py)
- Code symbols exposed: [`get_db_connection`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L7), [`init_db`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L14), [`add_student`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L44), [`get_all_students`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L75), [`mark_attendance`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L108), [`get_attendance_by_date`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L137), [`already_marked_today`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L174)
- Connection to other files: Provides the main database persistence backend for the enrollment scripts, recognizer cache, and Streamlit dashboards.

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`get_db_connection`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L7) | None | `sqlite3.Connection` | Opens an SQLite connection with active foreign key constraints and row dictionaries. |
| [`init_db`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L14) | None | None | Generates the `students` and `attendance` tables if they do not exist. |
| [`add_student`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L44) | `name: str`, `roll_number: str`, `encoding: np.ndarray` | `bool` | Registers a student profile and dumps their face signature vector to a binary BLOB. |
| [`get_all_students`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L75) | None | `list[dict]` | Returns a list of registered student records, deserializing face vectors from binary BLOBs. |
| [`mark_attendance`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L108) | `student_id: int`, `date: str`, `time: str`, `status: str` | `bool` | Inserts an attendance entry into the log table. |
| [`get_attendance_by_date`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L137) | `date: str` | `list[dict]` | Returns student attendance details for a targeted date by joining tables. |
| [`already_marked_today`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py#L174) | `student_id: int`, `date: str` | `bool` | Checks if a student has an attendance entry recorded on the given date. |

### 4.3. [src/enroll.py](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py)
This module registers a student by acquiring 10 frames of diverse poses from the webcam, processing their vectors, computing the average vector, and persisting it to the database. Includes a simulation fallback using mock matrices.
- Imports and dependencies: `os`, `sys`, `time`, `argparse`, `pathlib.Path`, `numpy` (as `np`), `PIL.Image`, `PIL.ImageDraw`, `cv2`, `face_recognition`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py)
- Code symbols exposed: `INSTRUCTIONS`, [`check_roll_number_exists`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L27), [`simulate_enrollment`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L32), [`enroll_student`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L85), [`cleanup_folder`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L240)
- Connection to other files: Executes enrollment tasks and updates `dataset/`. Integrated into the "Enroll Student" page of [`app/main.py`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py).

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`check_roll_number_exists`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L27) | `roll_number: str` | `bool` | Queries database to verify if a student roll number is already registered. |
| [`simulate_enrollment`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L32) | `name: str`, `roll_number: str`, `student_dir: Path` | `bool` | Generates placeholder images and a mock random normalized vector for webcam-free testing. |
| [`enroll_student`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L85) | `name: str`, `roll_number: str`, `simulate: bool` | `bool` | Controls webcam captures, guides the user through poses, averages face features, and adds student profiles. |
| [`cleanup_folder`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py#L240) | `folder_path: Path` | None | Removes image files and directories if enrollment is aborted. |

### 4.4. [src/enroll_from_folder.py](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py)
This script processes image datasets located in subfolders of `dataset/` named `<Name>_<RollNumber>`. It extracts face embeddings, averages them, and registers them in the database.
- Imports and dependencies: `os`, `sys`, `argparse`, `pathlib.Path`, `numpy` (as `np`), `face_recognition`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py)
- Code symbols exposed: [`check_roll_number_exists`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L12), [`enroll_student_from_directory`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L17), [`scan_and_enroll_dataset`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L112)
- Connection to other files: Interacts with the database backend to import students in bulk.

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`check_roll_number_exists`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L12) | `roll_number: str` | `bool` | Checks if a roll number is already present in the database. |
| [`enroll_student_from_directory`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L17) | `folder_path: Path`, `simulate: bool` | `bool` | Reads files in a folder, extracts face encodings, averages them, and registers the student. |
| [`scan_and_enroll_dataset`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py#L112) | `simulate: bool` | None | Scans the base directory for valid student folders and processes them sequentially. |

### 4.5. [src/recognizer.py](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py)
This module acts as the system matching engine. It extracts query vectors, caches database signatures, and computes similarity distances.
- Imports and dependencies: `sys`, `pathlib.Path`, `numpy` (as `np`), `face_recognition`, `cv2`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py)
- Code symbols exposed: `_KNOWN_ENCODINGS`, `_KNOWN_METADATA`, [`load_known_encodings`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py#L14), [`recognize_faces`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py#L45)
- Connection to other files: Provides the core face recognition engine for [`src/attendance.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py) and the Streamlit dashboard in [`app/main.py`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py).

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`load_known_encodings`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py#L14) | `force_reload: bool` | `tuple[list, list]` | Pulls student signatures from the database and updates the global cache. |
| [`recognize_faces`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py#L45) | `frame: np.ndarray`, `threshold: float` | `list[dict]` | Detects and identifies faces in a frame using Euclidean distances. |

### 4.6. [src/attendance.py](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py)
This module runs live attendance sessions. It processes camera frames, tracks verified students with debounce filters, records logs, and updates status values.
- Imports and dependencies: `os`, `sys`, `time`, `argparse`, `datetime.datetime`, `pathlib.Path`, `cv2`, `random`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py), [`src/recognizer.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py)
- Code symbols exposed: [`write_log`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L19), [`mark_absent_students`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L30), [`run_attendance_session`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L68), [`run_simulated_session`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L213)
- Connection to other files: Feeds visual data into [`app/main.py`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py) and records events in `logs/attendance_log.txt`.

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`write_log`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L19) | `message: str` | None | Appends timestamped log text to `logs/attendance_log.txt`. |
| [`mark_absent_students`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L30) | `date_str: str` | `int` | Identifies and logs absent students at session termination. |
| [`run_attendance_session`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L68) | `simulate: bool` | None | Coordinates video processing, database writes, visual overlays, and session finalization. |
| [`run_simulated_session`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py#L213) | `roll_to_student_map: dict` | None | Emulates real-time face matches and logs attendance in headless mode. |

### 4.7. [src/live_recognition_test.py](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py)
This module acts as a testing sandbox to verify camera streaming, face detections, and display overlays. It does not perform database inserts.
- Imports and dependencies: `sys`, `time`, `argparse`, `pathlib.Path`, `cv2`, `face_recognition`, `random`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py), [`src/recognizer.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py)
- Code symbols exposed: [`run_live_recognition`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py#L12), [`run_simulated_recognition_loop`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py#L109)
- Connection to other files: Uses the recognizer engine to run match queries for debugging.

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`run_live_recognition`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py#L12) | `simulate: bool` | None | Loops camera captures and displays bounding boxes. |
| [`run_simulated_recognition_loop`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py#L109) | `known_metadata: list` | None | Simulates detection frames within the command prompt window. |

### 4.8. [app/main.py](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py)
This is the entrypoint for the Streamlit dashboard. It renders enrollment screens, registers sessions, generates data visualizations, and exports logs as CSV files.
- Imports and dependencies: `sys`, `time`, `datetime.datetime`, `pathlib.Path`, `pandas` (as `pd`), `streamlit` (as `st`), `cv2`, `numpy` (as `np`), `face_recognition`, `PIL.Image`, `PIL.ImageDraw`, [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py), [`src/enroll.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py), [`src/attendance.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py), [`src/recognizer.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py)
- Code symbols exposed: [`show_marked_table`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py#L253)
- Connection to other files: Consolidates modules into a single web application wrapper.

#### Functions and Methods

| Function/Class | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| [`show_marked_table`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py#L253) | None | None | Renders a table of students marked present today inside the layout columns. |

### 4.9. [requirements.txt](file:///D:/Workspace/datascience-projects/face_attendance/requirements.txt)
Declarative file listing all necessary library dependencies and compiler tools.
- Imports and dependencies: None
- Code symbols exposed: None
- Connection to other files: Establishes the virtual environment dependencies.

#### Functions and Methods
Not applicable: This is a dependency manifest; it does not contain functional code.

### 4.10. [.gitignore](file:///D:/Workspace/datascience-projects/face_attendance/.gitignore)
Declarative file preventing temporary build files, databases, logs, and environments from being committed.
- Imports and dependencies: None
- Code symbols exposed: None
- Connection to other files: Configures git version control exclusions.

#### Functions and Methods
Not applicable: This is a configuration file; it does not contain functional code.

---

## 5. Setup Instructions

### Prerequisites

| Requirement | Minimum Version | Install Command |
| :--- | :--- | :--- |
| Python | 3.8 | Download and install from [python.org](https://www.python.org/) |
| PIP | 20.0 | `python -m pip install --upgrade pip` |
| SQLite3 | 3.0 | Built-in to Python Standard Library |
| CMake | 3.18 | Required if compiling `dlib` manually (`pip install cmake`) |
| VS C++ Build Tools | 2019 | Required if compiling `dlib` manually (via Visual Studio Installer) |

### Installation Steps

1. Open PowerShell or Command Prompt and clone or navigate to the project directory:
   ```powershell
   cd D:\Workspace\datascience-projects\face_attendance
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On Windows (Command Prompt):
     ```cmd
     .\venv\Scripts\activate.bat
     ```

4. Install the base libraries:
   ```powershell
   pip install -r requirements.txt
   ```

5. Install the pre-compiled `dlib` binary and the `face_recognition` library:
   ```powershell
   pip install dlib-bin
   pip install face_recognition --no-deps
   ```

6. Initialize the SQLite database tables:
   ```powershell
   python -m src.database
   ```

---

## 6. How to Run

### Running the Complete Application

To launch the web interface, execute the following command:
```powershell
python -m streamlit run app/main.py
```
This runs the web server and opens the dashboard in the default browser.

### Running Individual Modules

| File | Command | Expected Output/Behavior |
| :--- | :--- | :--- |
| [`app/main.py`](file:///D:/Workspace/datascience-projects/face_attendance/app/main.py) | `python -m streamlit run app/main.py` | Starts the web server and opens the Streamlit graphical panel. |
| [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py) | `python -m src.database` | Creates database tables and confirms file path initialization output. |
| [`src/enroll.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py) | `python -m src.enroll` | Prompts for name and roll number in console, opens webcam, and captures 10 poses. |
| [`src/enroll.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll.py) | `python -m src.enroll --simulate` | Performs student enrollment without camera using generated dummy files and vectors. |
| [`src/enroll_from_folder.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py) | `python -m src.enroll_from_folder` | Scans all child directories of `dataset/` and registers them. |
| [`src/enroll_from_folder.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/enroll_from_folder.py) | `python -m src.enroll_from_folder --simulate` | Registers folder students using dummy 128D vectors. |
| [`src/live_recognition_test.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py) | `python -m src.live_recognition_test` | Opens webcam frame output rendering face rectangles without log inserts. |
| [`src/live_recognition_test.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/live_recognition_test.py) | `python -m src.live_recognition_test --simulate` | Runs a console text output simulating random student captures. |
| [`src/attendance.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py) | `python -m src.attendance` | Launches webcam-based live session logging presence, auto-marks absent profiles upon exit. |
| [`src/attendance.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/attendance.py) | `python -m src.attendance --simulate` | Loops through random simulated detections in console and auto-finalizes absent entries. |

---

## 7. Configuration

Configurations are defined as constants inside [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py).

| Variable | Default | Description | Required (Y/N) |
| :--- | :--- | :--- | :--- |
| `BASE_DIR` | Absolute path to project root | Reference directory for all local module paths | Y |
| `DATASET_DIR` | `BASE_DIR / "dataset"` | Folder containing subdirectories with raw student images | Y |
| `ENCODINGS_DIR` | `BASE_DIR / "encodings"` | Storage directory for serialized signatures | Y |
| `DATABASE_DIR` | `BASE_DIR / "database"` | Path containing database files | Y |
| `LOGS_DIR` | `BASE_DIR / "logs"` | Path holding text-based log outputs | Y |
| `DOCS_DIR` | `BASE_DIR / "docs"` | Target folder for project documentation files | Y |
| `APP_DIR` | `BASE_DIR / "app"` | Source path for Streamlit dashboard assets | Y |
| `DB_PATH` | `DATABASE_DIR / "attendance.db"` | SQLite database file destination | Y |
| `RECOGNITION_THRESHOLD` | `0.6` | Metric distance for dlib matcher (smaller matches are stricter) | Y |
| `CAMERA_INDEX` | `0` | Camera capture hardware identifier | Y |

---

## 8. Testing

The system does not include unit testing frameworks. Testing is performed via simulation modes and standalone validation modules:

1. Live Webcam Testing:
   - Run command: `python -m src.live_recognition_test`
   - Description: Validates webcam streaming, face detection coordinates, bounding box renders, and cached matches.
2. Console Simulation Testing:
   - Run command: `python -m src.attendance --simulate`
   - Description: Tests the database insert loops, debounce timings, student dictionary queries, and automated absentee markings.
3. Streamlit Simulation Testing:
   - Action: Check the "Simulate" option in the Streamlit application for enrollment or attendance recording.
   - Description: Verifies web application flow, state transitions, table visualizations, and CSV exports.

---

## 9. Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ImportError: No module named cv2` | OpenCV is not installed. | Execute `pip install opencv-python`. |
| `dlib` fails to compile during install | Lack of C++ build compilers or CMake. | Run `pip install dlib-bin` to load precompiled binaries. |
| Webcam is black or fails to open | Incorrect index is specified or hardware permission is blocked. | Verify webcam connections. Change `CAMERA_INDEX` in [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py) to another index (e.g. `1`). |
| Multi-person detection errors | Multiple faces are present during enrollment. | Ensure that only the target student is positioned within the camera view. |
| Student recognized as "Unknown" | Lighting variations, poor angles, or strict threshold. | Adjust `RECOGNITION_THRESHOLD` in [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py) to a higher value (e.g. `0.65`). |
| Database locked error | Multiple processes modifying the database. | Close secondary execution processes to avoid SQLite write lock conflicts. |

---

## 10. Notes / Limitations

- HOG Model Limits: The HOG face detector is optimized for CPU speeds but struggles with non-frontal faces or profile angles. Deep CNN face models are more accurate but require dedicated CUDA-compatible GPUs.
- Access Control: The Streamlit interface lacks authentication, meaning any user can access configuration parameters, enrollments, and reports.
- Local SQLite Locking: SQLite limits concurrent write operations. Network deployments with multiple visual clients would require an enterprise server database (e.g., PostgreSQL).
- Serialization Constraints: Serializing arrays using python `pickle` creates vulnerability to python version updates. Storing signatures as structured JSON floats is recommended for upgrades.
- Spoofing Risks: The 2D webcam input can be spoofed using physical photographs or mobile screens showing enrolled faces. Liveness detection checks are required for security environments.
