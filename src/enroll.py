import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

# Import modules from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CAMERA_INDEX, DATASET_DIR
from src.database import add_student, get_all_students

# List of guide instructions for face enrollment steps
INSTRUCTIONS = [
    "Look straight at the camera",
    "Look straight at the camera",
    "Turn slightly left",
    "Turn slightly left",
    "Turn slightly right",
    "Turn slightly right",
    "Tilt head slightly up",
    "Tilt head slightly down",
    "Smile slightly",
    "Look straight and blink"
]

def check_roll_number_exists(roll_number: str) -> bool:
    """Checks if a roll number is already registered in the database."""
    students = get_all_students()
    return any(s['roll_number'] == roll_number for s in students)

def simulate_enrollment(name: str, roll_number: str, student_dir: Path) -> bool:
    """
    Simulates student enrollment by generating placeholder images 
    and a mock 128D face encoding. Useful in headless environments or when webcam is missing.
    """
    print(f"\n[SIMULATION] Starting mock enrollment for {name} (Roll: {roll_number})...")
    student_dir.mkdir(parents=True, exist_ok=True)
    
    # Save 10 placeholder images with labels
    for i in range(1, 11):
        img_path = student_dir / f"img_{i}.jpg"
        # Create a simple color block image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = (180, 150, 120)  # Muted brown/gray background
        
        # Draw some fake face outlines and text
        cv2.circle(img, (150, 130), 60, (255, 255, 255), 2)  # head outline
        cv2.circle(img, (130, 120), 8, (255, 255, 255), -1)   # left eye
        cv2.circle(img, (170, 120), 8, (255, 255, 255), -1)   # right eye
        cv2.ellipse(img, (150, 160), (25, 10), 0, 0, 180, (255, 255, 255), 2)  # smile
        
        cv2.putText(img, f"Simulated Face #{i}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Roll: {roll_number}", (30, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite(str(img_path), img)
        print(f"  -> Saved placeholder: dataset/{student_dir.name}/{img_path.name}")
        time.sleep(0.05)
        
    # Generate a random unit vector for mock 128D encoding
    mock_encoding = np.random.normal(0.0, 0.1, 128)
    mock_encoding /= np.linalg.norm(mock_encoding)
    
    # Insert student into DB
    success = add_student(name, roll_number, mock_encoding)
    if success:
        print(f"[SIMULATION] Success: Enrolled {name} ({roll_number}) with mock encoding.")
        return True
    else:
        print(f"[SIMULATION] Failure: Could not add {name} to the database.")
        return False

def enroll_student(name: str, roll_number: str, simulate: bool = False) -> bool:
    """
    Registers a new student by capturing webcam frames, extracting face encodings, 
    averaging them, and saving the record to the SQLite database.
    """
    # Sanitize and prepare paths
    sanitized_name = name.replace(" ", "_")
    student_dir = DATASET_DIR / f"{sanitized_name}_{roll_number}"
    
    # Error checking: Duplicate roll number
    if check_roll_number_exists(roll_number):
        print(f"\n[ERROR] Roll number '{roll_number}' is already registered!")
        return False
        
    if simulate:
        return simulate_enrollment(name, roll_number, student_dir)
        
    # Lazy import of face_recognition to allow simulation mode even if library is not installed
    try:
        import face_recognition
    except ImportError:
        print("\n[ERROR] 'face_recognition' library is not installed.")
        print("Please install requirements or run with --simulate to mock enrollment.")
        sys.exit(1)
        
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n[WARNING] Webcam index {CAMERA_INDEX} not accessible.")
        ans = input("Would you like to simulate enrollment with mock data instead? (y/n): ").strip().lower()
        if ans == 'y':
            return simulate_enrollment(name, roll_number, student_dir)
        return False

    print(f"\n[INFO] Starting enrollment for {name} ({roll_number}).")
    print("[INFO] Position yourself in front of the camera. Press 'q' or 'ESC' to abort.")
    
    student_dir.mkdir(parents=True, exist_ok=True)
    
    encodings_list = []
    saved_images_paths = []
    step = 0
    cooldown_duration = 2.0  # seconds between captures
    cooldown_start = time.time()
    warning_msg = None
    
    while step < len(INSTRUCTIONS):
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break
            
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Real-time face detection
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw bounding boxes
        for top, right, bottom, left in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
        # Analyze face count
        if len(face_locations) == 0:
            warning_msg = "No face detected! Please position your face in the frame."
        elif len(face_locations) > 1:
            warning_msg = "Multiple faces detected! Make sure only one person is in view."
        else:
            warning_msg = None
            
            # Check capture timer
            elapsed = time.time() - cooldown_start
            if elapsed >= cooldown_duration:
                # Capture frame
                img_path = student_dir / f"img_{step + 1}.jpg"
                cv2.imwrite(str(img_path), frame)
                saved_images_paths.append(img_path)
                
                # Compute face encoding
                encs = face_recognition.face_encodings(rgb_frame, face_locations)
                if encs:
                    encodings_list.append(encs[0])
                    print(f"Captured {step + 1}/{len(INSTRUCTIONS)} - Image saved to {img_path.name}")
                    step += 1
                    cooldown_start = time.time()
                else:
                    warning_msg = "Could not parse facial features. Retrying..."
                    # Clean up file if encoding failed
                    if img_path.exists():
                        img_path.unlink()
                        
        # UI Overlays
        # Instruction and status bottom bar
        cv2.rectangle(display_frame, (0, h - 80), (w, h), (30, 30, 30), -1)
        
        if step < len(INSTRUCTIONS):
            cv2.putText(display_frame, f"Step {step+1}/{len(INSTRUCTIONS)}: {INSTRUCTIONS[step]}", 
                        (20, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            time_left = max(0.0, cooldown_duration - (time.time() - cooldown_start))
            if warning_msg:
                cv2.putText(display_frame, f"STATUS: {warning_msg}", 
                            (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, f"Capturing next pose in {time_left:.1f}s", 
                            (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
        # Urgent warning top bar if multiple faces
        if len(face_locations) > 1:
            cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 150), -1)
            cv2.putText(display_frame, "WARNING: MULTIPLE PEOPLE DETECTED!", 
                        (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Student Enrollment", display_frame)
        
        # Check keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("[INFO] Enrollment process canceled by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Process results
    if step == len(INSTRUCTIONS) and len(encodings_list) > 0:
        print("[INFO] Encodings captured. Calculating average face signature...")
        # Average encodings to get a single unified 128D encoding
        avg_encoding = np.mean(encodings_list, axis=0)
        
        # Save to database
        success = add_student(name, roll_number, avg_encoding)
        if success:
            print(f"[SUCCESS] Student '{name}' (Roll: {roll_number}) enrolled successfully!")
            return True
        else:
            print("[ERROR] Failed to save student to database.")
            # Rollback file writes
            cleanup_folder(student_dir)
            return False
    else:
        print("[WARNING] Enrollment was incomplete. Cleaning up files...")
        cleanup_folder(student_dir)
        return False

def cleanup_folder(folder_path: Path):
    """Safely removes a folder and all its contents if enrollment fails/aborts."""
    if folder_path.exists() and folder_path.is_dir():
        for file in folder_path.iterdir():
            try:
                file.unlink()
            except Exception as e:
                print(f"Could not delete {file}: {e}")
        try:
            folder_path.rmdir()
        except Exception as e:
            print(f"Could not delete directory {folder_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll student into Face Recognition Database.")
    parser.add_argument("--simulate", action="store_true", help="Simulate enrollment using dummy data without webcam.")
    args = parser.parse_argument_group().parser.parse_args()  # Simple parsing
    
    print("=== Student Face Enrollment Module ===")
    name_input = input("Enter Student Full Name: ").strip()
    roll_input = input("Enter Student Roll Number/ID: ").strip()
    
    if not name_input or not roll_input:
        print("[ERROR] Name and Roll Number are required fields!")
        sys.exit(1)
        
    enroll_student(name_input, roll_input, simulate=args.simulate)
