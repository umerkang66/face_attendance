# Troubleshooting and FAQ Guide

This document lists common issues, build errors, hardware difficulties, and database locks encountered during setup or runtime, along with their solutions.

---

## 1. Installation and Build Issues

### Issue: `dlib` compilation failures during `pip install`

- **Symptoms:** The installation process hangs or exits with errors related to missing C++ compilers, `cl.exe`, or CMake parameters.
- **Root Cause:** Compiling `dlib` from source requires local CMake installations and Visual Studio C++ build toolchains.
- **Resolution:** Install precompiled binaries using `dlib-bin` to skip local compiling steps:
  ```powershell
  pip install dlib-bin
  ```

### Issue: `ImportError: No module named cv2`

- **Symptoms:** The python script crashes immediately on launch with this module import error.
- **Root Cause:** OpenCV is missing from the active virtual environment.
- **Resolution:** Install the OpenCV wrapper package:
  ```powershell
  pip install opencv-python
  ```

### Issue: `ImportError: libGL.so.1: cannot open shared object file` or missing C++ compiler on Linux

- **Symptoms:** On Linux servers, Docker, or Streamlit Cloud, OpenCV throws `libGL.so.1` or `libglib-2.0.so.0` errors, or `dlib` build fails due to missing `cmake`/`g++`.
- **Root Cause:** Operating system lacks necessary C++ build tools and OpenGL runtime packages.
- **Resolution:** Install the APT packages listed in [`package.txt`](../package.txt):
  ```bash
  sudo apt-get update && sudo apt-get install -y $(cat package.txt)
  ```
  _(Note: Streamlit Community Cloud reads `package.txt` automatically during deployment)._

---

## 2. Hardware and Camera Issues

### Issue: The webcam window is black or fails to open

- **Symptoms:** The script outputs camera opening errors or launches a blank black GUI window.
- **Root Cause:** The default camera index is incorrect or the operating system is blocking webcam access.
- **Resolution:**
  1. Open [`src/config.py`](../src/config.py) and change the `CAMERA_INDEX` (e.g., set `CAMERA_INDEX = 1` or `2` for external USB webcams).
  2. Verify that other applications (such as Zoom, Teams, or Skype) are not actively locking the webcam.
  3. Check system privacy settings to ensure the terminal/app is permitted to access the camera hardware.

---

## 3. Database Issues

### Issue: `sqlite3.OperationalError: database is locked`

- **Symptoms:** The application throws exception traces during attendance logging or student enrollment actions.
- **Root Cause:** SQLite3 locks database writes at the file level. Concurrent processes (e.g., Streamlit dashboard and a standalone enrollment terminal) are attempting to write to [`database/attendance.db`](../database/attendance.db) simultaneously.
- **Resolution:**
  1. Ensure only one active application is executing database writes at any time.
  2. Close any lingering background processes or terminals running the system scripts.

---

## 4. Recognition and Accuracy Issues

### Issue: Enrolled students are recognized as "Unknown"

- **Symptoms:** The live video view displays red bounding boxes for registered student faces.
- **Root Cause:** The Euclidean distance calculation exceeds the default `RECOGNITION_THRESHOLD` (0.6) due to extreme lighting changes, off-angle head positioning, or poor initial enrollment captures.
- **Resolution:**
  1. Increase the threshold slightly in [`src/config.py`](../src/config.py) (e.g., set `RECOGNITION_THRESHOLD = 0.65`). Note that higher values increase false-acceptance rates.
  2. Re-enroll the student under clean, even lighting conditions, ensuring they capture multiple facial angles.

### Issue: Multiple faces detected during enrollment

- **Symptoms:** Enrollment captures fail, displaying warning messages or returning mismatched average vectors.
- **Root Cause:** The HOG detector is picking up background faces or reflections in the frame.
- **Resolution:** Ensure that only the target student is positioned within the camera view during the enrollment process.
