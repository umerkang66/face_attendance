# Face Recognition Attendance Management System

This is an automated student attendance tracking system using computer vision. The system captures face images, computes 128D embeddings using dlib-based facial signatures, manages student records in an SQLite database, and offers a premium Streamlit control panel interface alongside standalone command-line scripts.

---

## 🚀 Key Features

- **Premium Streamlit Dashboard (`app/main.py`):**
  - **Student Registration:** Register students by capturing 10 facial poses under guided visual timers or simulating enrollment with mock profiles.
  - **Live Attendance Recording:** Match live camera streams against registered database embeddings in real-time, leveraging debounce protection to avoid duplicate scans.
  - **Analytics Reports:** Monitor today's attendance summary stats (Present, Absent, total Class Strength), inspect visual charts, browse student logs, and export reports to CSV format.
- **Flexible Operational Modes:** Support for both standard webcam-based operation and full console/UI simulation (highly useful for headless servers or local development without camera access).
- **Bulk Folder Enrollment:** Easily ingest pre-existing student photo directories matching specific naming conventions.
- **Automatic Absentee Marking:** Automatically detects unmarked students at the end of a session and flags them as "Absent".
- **Logging System:** Keeps timestamped records of all marked attendance events in a text file.

---

## 🛠️ Tech Stack

- **Core Language:** Python 3.x
- **Computer Vision Frameworks:** OpenCV (`opencv-python`), dlib-based embeddings (`face_recognition`)
- **Database Management:** SQLite3 (standard library)
- **Dashboard & UI:** Streamlit
- **Data Processing & Visualization:** Pandas, NumPy
- **Image Processing:** Pillow (PIL)

---

## 📂 Project Structure

```text
face_attendance/
├── app/
│   └── main.py              # Streamlit dashboard and UI control panel
├── database/
│   └── attendance.db        # SQLite database storing students & attendance logs
├── dataset/                 # Enrolled student face images grouped by folders
├── docs/                    # Project documentation and report materials
├── encodings/               # Directory for serialized face embeddings (if exported)
├── logs/
│   └── attendance_log.txt   # File-based logging for attendance operations
├── src/
│   ├── config.py            # Global directories, recognition thresholds, camera settings
│   ├── database.py          # SQLite database schema initialization and queries
│   ├── enroll.py            # Script for enrolling single students via webcam/simulation
│   ├── enroll_from_folder.py# Script for bulk enrolling students from subfolders
│   ├── attendance.py        # Logic for attendance sessions (live and console simulation)
│   ├── recognizer.py        # Core face detection, encoding, matching, and threshold logic
│   └── live_recognition_test.py # Live facial recognition visual test sandbox
├── requirements.txt         # Package dependencies
└── README.md                # System documentation (this file)
```

---

## 🗄️ Database Schema

The database consists of two tables defined in [`src/database.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/database.py):

### 1. `students`

Stores registered student metadata and dlib facial signatures.

- `id` (INTEGER, Primary Key, Autoincrement)
- `name` (TEXT, Not Null)
- `roll_number` (TEXT, Unique, Not Null)
- `encoding` (BLOB, Serialized 128D NumPy array via `pickle`)
- `date_enrolled` (TEXT, Default `CURRENT_TIMESTAMP`)

### 2. `attendance`

Tracks daily attendance entries.

- `id` (INTEGER, Primary Key, Autoincrement)
- `student_id` (INTEGER, Foreign Key referencing `students(id)` on cascade delete)
- `date` (TEXT, Not Null, Format: `YYYY-MM-DD`)
- `time` (TEXT, Not Null, Format: `HH:MM:SS`)
- `status` (TEXT, Not Null, e.g., `'Present'`, `'Absent'`)

---

## ⚙️ Installation & Setup

Follow these steps to set up the environment and run the system on Windows:

### 1. Install System Dependencies

Install python dependencies from the requirements manifest:

```bash
pip install -r requirements.txt
```

### 2. Install Face Recognition Libraries

Installing `dlib` on Windows can sometimes require CMake and Visual Studio compiler setups. You can bypass local compilation by installing pre-compiled binaries:

```bash
# Install pre-compiled dlib binary
pip install dlib-bin

# Install face_recognition avoiding circular dependency conflicts
pip install face_recognition --no-deps
```

### 3. Initialize the SQLite Database

Set up the SQLite database file and tables:

```bash
python -m src.database
```

This will initialize [`database/attendance.db`](file:///D:/Workspace/datascience-projects/face_attendance/database/attendance.db) with the correct table schemas.

---

## 🖥️ Running the Application

### A. Main Dashboard (Recommended)

Launch the Streamlit web dashboard:

```bash
python -m streamlit run app/main.py
```

This launches a browser window with access to Student Registration, live attendance taking, and analytics reporting.

### B. Command-Line Interface (CLI) Commands

#### 1. Single Student Enrollment

Register a single student via command prompt:

```bash
# Standard webcam enrollment (opens camera, guides user through 10 poses)
python -m src.enroll

# Simulation mode (head-free environment, creates dummy student images and encoding)
python -m src.enroll --simulate
```

#### 2. Bulk Enrollment from Folders

Register multiple students who have folders of images saved under `dataset/` (folders must follow the `<Name>_<RollNumber>` structure, e.g., `dataset/john_doe_101/`):

```bash
# Process existing dataset directories
python -m src.enroll_from_folder

# Mock dataset directory registration (generates mock encodings without face_recognition)
python -m src.enroll_from_folder --simulate
```

#### 3. Live Recognition Visual Sandbox

Test the camera matches without recording attendance metrics:

```bash
# Standard webcam matching
python -m src.live_recognition_test

# Test console-based simulation loops
python -m src.live_recognition_test --simulate
```

#### 4. Console-Based Attendance Session

Record attendance via terminal session:

```bash
# Standard webcam attendance tracking
python -m src.attendance

# Run mock terminal session matching random students
python -m src.attendance --simulate
```

---

## 🔍 Face Recognition Performance Details

As documented in [`src/recognizer.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/recognizer.py), the system uses dlib's 128D face mapping. The similarity threshold (`RECOGNITION_THRESHOLD` in [`src/config.py`](file:///D:/Workspace/datascience-projects/face_attendance/src/config.py), set to `0.6` by default) balances the performance:

- **Low Threshold (e.g. `0.4`):** Strictly rejects strangers (low False-Accept Rate) but might miss students under poor lighting (high False-Reject Rate).
- **Medium Threshold (`0.6`):** Balanced operating point. Highly accurate verification with normal tolerance for expression/lighting changes.
- **High Threshold (e.g. `0.8`):** Lenient matching. Easy verification but risks false identification (high False-Accept Rate).
