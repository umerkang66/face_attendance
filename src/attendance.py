import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Import modules from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CAMERA_INDEX, LOGS_DIR
from src.database import (
    get_all_students, 
    mark_attendance, 
    already_marked_today, 
    get_attendance_by_date
)
from src.recognizer import recognize_faces, load_known_encodings

def write_log(message: str):
    """Writes a timestamped message to the attendance log file."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "attendance_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to log file: {e}")

def mark_absent_students(date_str: str = None) -> int:
    """
    Compares the full student enrollment list against today's attendance logs
    and automatically marks all unrecorded students as "Absent".
    
    Args:
        date_str: Target date string (YYYY-MM-DD). Defaults to today's date.
        
    Returns:
        The number of students newly marked as Absent.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    print(f"\n[FINALIZING] Finalizing attendance for {date_str}...")
    
    all_students = get_all_students()
    current_attendance = get_attendance_by_date(date_str)
    
    # Get IDs of all students already marked today (either Present, Absent, etc.)
    recorded_student_ids = {record['student_id'] for record in current_attendance}
    
    absent_count = 0
    time_str = "00:00:00"  # Default time stamp for automated entries
    
    for student in all_students:
        student_id = student['id']
        if student_id not in recorded_student_ids:
            success = mark_attendance(student_id, date_str, time_str, "Absent")
            if success:
                msg = f"Auto-Finalize: Marked '{student['name']}' (Roll: {student['roll_number']}) as ABSENT"
                print(f"  -> {msg}")
                write_log(msg)
                absent_count += 1
                
    print(f"[FINALIZING] Session finalized. {absent_count} student(s) marked Absent.")
    return absent_count

def run_attendance_session(simulate: bool = False):
    """
    Main real-time loops that coordinates the webcam feed, runs recognition,
    marks present students with debouncing, draws overlay cues, and finalizes 
    unmarked students as absent upon termination.
    """
    print("=== Starting Attendance Session ===")
    
    # Reload known face signatures
    known_encodings, _ = load_known_encodings(force_reload=True)
    if not known_encodings:
        print("[WARNING] Face database is empty. No faces will be recognized.")
        print("Please enroll students first.")
        
    # Get a mapping of Roll Number to Student Database ID for fast lookup
    all_students = get_all_students()
    roll_to_student_map = {s['roll_number']: s for s in all_students}
    
    if simulate:
        run_simulated_session(roll_to_student_map)
        return
        
    # Lazy imports of CV libraries
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] 'opencv-python' (cv2) library is not installed.")
        print("Please install requirements or run with --simulate to run a console test.")
        sys.exit(1)
        
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n[WARNING] Webcam index {CAMERA_INDEX} is not accessible.")
        ans = input("Would you like to run in console-based simulation mode instead? (y/n): ").strip().lower()
        if ans == 'y':
            run_simulated_session(roll_to_student_map)
        return
        
    print(f"\n[INFO] Session Active. Webcam index: {CAMERA_INDEX}")
    print("[INFO] Controls:")
    print("   - Press 'q' or 'ESC' to end session (runs auto-absent finalized).")
    print("   - Press 'e' to trigger manual override end-of-session.")
    
    # State dictionaries for tracking
    session_cache = {}    # student_id -> last_checked_time (debounce lookup)
    newly_marked = {}     # student_id -> marked_timestamp (confirmation popup duration)
    
    debounce_duration = 3.0       # Re-check database every 3 seconds for the same face
    confirmation_duration = 2.0   # Keep the "Marked ✓" overlay visible for 2 seconds
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break
            
        h, w, _ = frame.shape
        
        # Run recognition
        results = recognize_faces(frame)
        now_time = time.time()
        
        for face in results:
            top, right, bottom, left = face['bbox']
            name = face['name']
            roll_number = face['roll_number']
            confidence = face['confidence']
            
            is_newly_marked = False
            is_present = False
            
            if name != "Unknown" and roll_number in roll_to_student_map:
                student = roll_to_student_map[roll_number]
                student_id = student['id']
                is_present = True
                
                # Apply debounce check
                last_check = session_cache.get(student_id, 0.0)
                if (now_time - last_check) > debounce_duration:
                    session_cache[student_id] = now_time
                    
                    # Query Database to check if already marked today
                    if not already_marked_today(student_id, today_date):
                        current_time = datetime.now().strftime("%H:%M:%S")
                        success = mark_attendance(student_id, today_date, current_time, "Present")
                        if success:
                            msg = f"Marked '{name}' (Roll: {roll_number}) as Present (Time: {current_time})"
                            print(f"[ATTENDANCE] {msg}")
                            write_log(msg)
                            newly_marked[student_id] = now_time
                            
                # Check if we should display the temporary confirmation overlay
                marked_time = newly_marked.get(student_id, 0.0)
                if (now_time - marked_time) < confirmation_duration:
                    is_newly_marked = True
            
            # Rendering drawing settings
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label_text = "Unknown"
            elif is_newly_marked:
                color = (0, 255, 0)  # Green for newly marked
                label_text = f"Marked! {name} ({confidence*100:.0f}%)"
            else:
                color = (255, 255, 0)  # Cyan/Yellow for already checked
                label_text = f"Present: {name} ({confidence*100:.0f}%)"
                
            # Draw Face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw Bottom Label Bar
            cv2.rectangle(frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label_text, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show additional checked badge top-left of box
            if name != "Unknown":
                badge = "Marked" if not is_newly_marked else "NEW ✓"
                cv2.putText(frame, badge, (left, top - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                
        # Draw session header info
        cv2.putText(frame, f"Session Date: {today_date} | Press 'e' or 'q' to end", 
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Attendance Session", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("[INFO] Ending session...")
            break
        elif key == ord('e'):
            print("[INFO] Manual finalization request received.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Auto-finalize session: Mark remaining students as Absent
    mark_absent_students(today_date)
    print("=== Attendance Session Completed ===")

def run_simulated_session(roll_to_student_map: dict):
    """Console-based mock session execution for terminal testing."""
    print("\n[SIMULATION] Launching simulated attendance session (Ctrl+C or let it finish)...")
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    students = list(roll_to_student_map.values())
    
    if not students:
        print("[SIMULATION] No registered students found in database. Cannot simulate markings.")
        return
        
    session_cache = {}
    newly_marked = {}
    
    print(f"[SIMULATION] Enrolled student bank size: {len(students)}")
    print("[SIMULATION] Running 5 mock camera frames...")
    
    import random
    
    try:
        for frame in range(1, 6):
            print(f"\n--- Simulated Frame {frame}/5 ---")
            # Randomly select a student to simulate detecting
            student = random.choice(students)
            student_id = student['id']
            name = student['name']
            roll_number = student['roll_number']
            
            print(f"[CAMERA] Detected face: {name} (Roll: {roll_number})")
            
            # Run debounce logic simulation
            now = time.time()
            last_check = session_cache.get(student_id, 0.0)
            if (now - last_check) > 3.0:
                session_cache[student_id] = now
                
                # Check DB status
                if not already_marked_today(student_id, today_date):
                    current_time = datetime.now().strftime("%H:%M:%S")
                    success = mark_attendance(student_id, today_date, current_time, "Present")
                    if success:
                        msg = f"Marked '{name}' (Roll: {roll_number}) as Present (Time: {current_time})"
                        print(f"  [ATTENDANCE] {msg}")
                        write_log(msg)
                        newly_marked[student_id] = now
                else:
                    print(f"  [DEBOUNCE] '{name}' already marked today. Database insert skipped.")
            else:
                print(f"  [DEBOUNCE] Debounce active for '{name}'. Skipping.")
                
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[SIMULATION] Simulation aborted by user.")
        
    # Auto-finalize absent students at the end
    mark_absent_students(today_date)
    print("\n=== Simulated Attendance Session Completed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Recognition Attendance Session.")
    parser.add_argument("--simulate", action="store_true", help="Simulate a live attendance session in console.")
    args = parser.parse_args()
    
    run_attendance_session(simulate=args.simulate)
