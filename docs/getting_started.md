# Getting Started Guide

This guide details the prerequisites, installation instructions, running procedures, and testing methods for the Face Recognition Attendance Management System.

---

## 1. Prerequisites

The system relies on Python and several key platform dependencies. Below is the minimum specification required:

| Dependency | Minimum Version | Installation / Configuration Method |
| :--- | :--- | :--- |
| Python | 3.8 | Download and install from [python.org](https://www.python.org/) |
| PIP | 20.0 | Run `python -m pip install --upgrade pip` |
| SQLite3 | 3.0 | Included in Python's Standard Library |
| CMake | 3.18 | Required if compiling `dlib` manually (`pip install cmake`) |
| VS C++ Build Tools | 2019 | Required if compiling `dlib` manually (via Visual Studio Installer) |

---

## 2. Installation Steps

Follow these steps to configure your local development environment:

### Step 1: Clone or Navigate to the Directory
Open PowerShell or Command Prompt and change directory to the project root:
```powershell
cd D:\Workspace\datascience-projects\face_attendance
```

### Step 2: Establish a Virtual Environment
Initialize a clean environment to manage dependencies:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
Activate the environment to isolate project libraries:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
Install the required packages from [`requirements.txt`](../requirements.txt):
```powershell
pip install -r requirements.txt
```

### Step 5: Install OpenCV, dlib, and face_recognition
Install the pre-compiled `dlib` binary and the `face_recognition` package without installing overlapping dependencies:
```powershell
pip install dlib-bin
pip install face_recognition --no-deps
```

### Step 6: Initialize Database Schema
Execute the database script to instantiate the SQLite tables:
```powershell
python -m src.database
```

---

## 3. How to Run

### Real-Time Web Interface (Streamlit Dashboard)
To run the primary user dashboard:
```powershell
python -m streamlit run app/main.py
```
This launches a local web server and opens the dashboard in your default browser.

### Executing Standalone Modules

| Script | Command | Behavior |
| :--- | :--- | :--- |
| [`app/main.py`](../app/main.py) | `python -m streamlit run app/main.py` | Launches the Streamlit control panel interface. |
| [`src/database.py`](../src/database.py) | `python -m src.database` | Creates database tables and initializes the DB file. |
| [`src/enroll.py`](../src/enroll.py) | `python -m src.enroll` | Collects student name/roll number and captures 10 webcam poses. |
| [`src/enroll.py`](../src/enroll.py) | `python -m src.enroll --simulate` | Executes enrollment without camera using random mock vectors. |
| [`src/enroll_from_folder.py`](../src/enroll_from_folder.py) | `python -m src.enroll_from_folder` | Scans the `dataset/` directory and registers students in bulk. |
| [`src/enroll_from_folder.py`](../src/enroll_from_folder.py) | `python -m src.enroll_from_folder --simulate` | Batch imports student directories using mock 128D encodings. |
| [`src/live_recognition_test.py`](../src/live_recognition_test.py) | `python -m src.live_recognition_test` | Validates face recognition loops on live camera frames. |
| [`src/live_recognition_test.py`](../src/live_recognition_test.py) | `python -m src.live_recognition_test --simulate` | Executes console-based recognition loops using mock signatures. |
| [`src/attendance.py`](../src/attendance.py) | `python -m src.attendance` | Runs camera session logging attendance; marks absentees on exit. |
| [`src/attendance.py`](../src/attendance.py) | `python -m src.attendance --simulate` | Loops simulated presence checks in terminal and registers records. |

---

## 4. Testing Procedures

The system utilizes specialized test modes to verify subsystem functionality:

### 1. Live Webcam and Computer Vision Verification
- **Command:** `python -m src.live_recognition_test`
- **Verification Details:** Validates video capturing frame rates, face boundary localization coordinates, and the cached matching lookup latency.

### 2. Headless Console Simulation
- **Command:** `python -m src.attendance --simulate`
- **Verification Details:** Bypasses visual hardware to execute database insert operations, debounce delays, dictionary lookups, and absentee automated markings.

### 3. Streamlit Interface Simulation
- **Action:** Toggle the "Simulate" option inside the Streamlit sidebars.
- **Verification Details:** Tests frontend dashboards, table refreshes, CSV record exports, and enrollment inputs.
