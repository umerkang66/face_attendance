import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Import modules from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATASET_DIR
from src.database import add_student, get_all_students

def check_roll_number_exists(roll_number: str) -> bool:
    """Checks if a roll number is already registered in the database."""
    students = get_all_students()
    return any(s['roll_number'] == roll_number for s in students)

def enroll_student_from_directory(folder_path: Path, simulate: bool = False) -> bool:
    """
    Processes all images inside a specific student's directory, extracts and averages face encodings,
    and inserts the student record into the SQLite database.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return False
        
    folder_name = folder_path.name
    
    # Parse name and roll number from directory name (Format: Name_RollNumber)
    if "_" not in folder_name:
        print(f"[ERROR] Folder name '{folder_name}' must be formatted as '<name>_<roll_number>' (e.g., 'john_doe_101'). Skipping.")
        return False
        
    parts = folder_name.rsplit('_', 1)
    name = parts[0].replace('_', ' ')
    roll_number = parts[1]
    
    # Check if duplicate roll number
    if check_roll_number_exists(roll_number):
        print(f"[SKIP] Student with Roll Number '{roll_number}' ({name}) is already registered.")
        return False
        
    print(f"\n[PROCESSING] Scanning folder '{folder_name}' for student '{name}' (Roll: {roll_number})...")
    
    # Find all common image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jpg'.upper(), '.jpeg'.upper(), '.png'.upper()}
    image_paths = [p for p in folder_path.iterdir() if p.suffix in image_extensions]
    
    if not image_paths:
        print(f"[WARNING] No image files found in '{folder_path}'. Expected .jpg, .jpeg, or .png. Skipping.")
        return False
        
    encodings_list = []
    
    if simulate:
        # Generate dummy 128D encodings in simulation mode
        print(f"  [SIMULATION] Simulating face detection on {len(image_paths)} images...")
        for img_path in image_paths:
            print(f"  -> Simulated face detection on: {img_path.name}")
            mock_enc = np.random.normal(0.0, 0.1, 128)
            mock_enc /= np.linalg.norm(mock_enc)
            encodings_list.append(mock_enc)
    else:
        # Lazy import of face_recognition
        try:
            import face_recognition
        except ImportError:
            print("[ERROR] 'face_recognition' library is not installed.")
            print("Please install requirements or run with --simulate to mock folder enrollment.")
            sys.exit(1)
            
        for img_path in image_paths:
            try:
                # Load image
                image = face_recognition.load_image_file(str(img_path))
                
                # Detect face locations
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    print(f"  [WARNING] No face detected in '{img_path.name}'. Skipping.")
                    continue
                elif len(face_locations) > 1:
                    print(f"  [WARNING] Multiple faces ({len(face_locations)}) detected in '{img_path.name}'. Skipping to prevent encoding pollution.")
                    continue
                    
                # Extract encoding
                encs = face_recognition.face_encodings(image, face_locations)
                if encs:
                    encodings_list.append(encs[0])
                    print(f"  [OK] Extracted face encoding from: {img_path.name}")
                else:
                    print(f"  [WARNING] Failed to compute encoding for '{img_path.name}'. Skipping.")
            except Exception as e:
                print(f"  [ERROR] Failed to process image '{img_path.name}': {e}")
                
    if not encodings_list:
        print(f"[ERROR] No valid face encodings could be extracted from '{folder_name}'. Student not enrolled.")
        return False
        
    # Average encodings to get a single unified 128D encoding signature
    avg_encoding = np.mean(encodings_list, axis=0)
    
    # Store to SQLite
    success = add_student(name, roll_number, avg_encoding)
    if success:
        print(f"[SUCCESS] Bulk-enrolled student: {name} (Roll: {roll_number}) with {len(encodings_list)} valid image source(s).")
        return True
    else:
        print(f"[ERROR] Could not save student {name} ({roll_number}) to the database.")
        return False

def scan_and_enroll_dataset(simulate: bool = False):
    """Scans all subfolders in the root dataset folder and registers them."""
    print("=== Bulk Folder Enrollment Module ===")
    print(f"Scanning base dataset directory: {DATASET_DIR}")
    
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory {DATASET_DIR} does not exist.")
        return
        
    subfolders = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    if not subfolders:
        print("[INFO] No subdirectories found in dataset folder.")
        return
        
    success_count = 0
    fail_count = 0
    
    for folder in subfolders:
        # Ignore pycache or hidden folders
        if folder.name.startswith('.') or folder.name.startswith('__'):
            continue
            
        result = enroll_student_from_directory(folder, simulate=simulate)
        if result:
            success_count += 1
        else:
            fail_count += 1
            
    print(f"\n=== Bulk Enrollment Summary ===")
    print(f"Successfully Enrolled: {success_count}")
    print(f"Skipped / Failed:      {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk enroll students from existing images organized in folders.")
    parser.add_argument("--folder", type=str, help="Path to a specific student directory (e.g. 'dataset/alice_007'). If not specified, scans the entire base dataset directory.")
    parser.add_argument("--simulate", action="store_true", help="Simulate face encoding extraction without loading image libraries.")
    args = parser.parse_argument_group().parser.parse_args()
    
    if args.folder:
        target_path = Path(args.folder).resolve()
        enroll_student_from_directory(target_path, simulate=args.simulate)
    else:
        scan_and_enroll_dataset(simulate=args.simulate)
