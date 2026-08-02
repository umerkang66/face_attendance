import sys
import time
import argparse
from pathlib import Path

# Import modules from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CAMERA_INDEX
from src.database import get_all_students
from src.recognizer import recognize_faces, load_known_encodings, draw_face_label

def run_live_recognition(simulate: bool = False):
    """
    Main loop for live face recognition. Opens the webcam, reads frames,
    detects & matches faces, draws boxes and labels, and shows the output.
    """
    print("=== Live Face Recognition Test ===")
    
    # Load registered students
    known_encodings, known_metadata = load_known_encodings(force_reload=True)
    if not known_encodings:
        print("[WARNING] Database is currently empty. All faces will be classified as 'Unknown'.")
        print("Please enroll students first using 'src/enroll.py' or 'src/enroll_from_folder.py'.")
        
    if simulate:
        run_simulated_recognition_loop(known_metadata)
        return
        
    # Lazy imports of CV libraries
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] 'opencv-python' (cv2) library is not installed.")
        print("Please install requirements or run with --simulate to test in console mode.")
        sys.exit(1)
        
    try:
        import face_recognition
    except ImportError:
        print("\n[ERROR] 'face_recognition' library is not installed.")
        print("Please install requirements or run with --simulate to test in console mode.")
        sys.exit(1)
        
    # Open webcam capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n[WARNING] Webcam index {CAMERA_INDEX} is not accessible.")
        ans = input("Would you like to run in console-based simulation mode instead? (y/n): ").strip().lower()
        if ans == 'y':
            run_simulated_recognition_loop(known_metadata)
        return
        
    print(f"\n[INFO] Initialized webcam camera index: {CAMERA_INDEX}")
    print("[INFO] Press 'q' or 'ESC' on the video window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam. Exiting loop.")
            break
            
        # Run recognition on the grabbed frame
        results = recognize_faces(frame)
        
        # Draw bounding boxes and text labels on the frame
        for face in results:
            top, right, bottom, left = face['bbox']
            name = face['name']
            roll_number = face['roll_number']
            confidence = face['confidence']
            
            # Choose color: Green for known, Red for unknown faces
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({confidence * 100:.0f}%)" if name != "Unknown" else "Unknown"
            top_badge = f"Roll: {roll_number}" if (name != "Unknown" and roll_number) else None
            
            draw_face_label(frame, (top, right, bottom, left), label, color=color, top_badge=top_badge)
                
        # Show number of faces in top corner
        status_text = f"Faces in frame: {len(results)}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Render video
        cv2.imshow("Live Face Recognition Test", frame)
        
        # Stop loop if 'q' or 'ESC' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    # Release assets
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live test completed successfully.")

def run_simulated_recognition_loop(known_metadata: list):
    """Console-based mock recognition loop for validation without GUI / webcam."""
    print("\n[SIMULATION] Starting console-based recognition loop simulation (Press Ctrl+C to stop)...")
    print("[SIMULATION] Fetching mock camera frames at 1-second intervals...")
    
    import random
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            # Randomly simulate detecting 0, 1, or 2 faces
            num_faces = random.choices([0, 1, 2], weights=[20, 60, 20])[0]
            
            simulated_matches = []
            for _ in range(num_faces):
                # 80% chance of recognizing an enrolled student, 20% chance of Unknown
                if known_metadata and random.random() < 0.8:
                    match = random.choice(known_metadata)
                    confidence = random.uniform(0.75, 0.98)
                    simulated_matches.append(f"{match['name']} ({match['roll_number']}) [{confidence * 100:.1f}% confidence]")
                else:
                    confidence = random.uniform(0.40, 0.55)
                    simulated_matches.append(f"Unknown [{confidence * 100:.1f}% confidence]")
            
            print(f"[Frame {frame_count:03d}] Detected {num_faces} face(s). Matches: {', '.join(simulated_matches) if simulated_matches else 'None'}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[SIMULATION] Simulated recognition loop terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face recognition live on webcam.")
    parser.add_argument("--simulate", action="store_true", help="Simulate face detection output in console without GUI.")
    args = parser.parse_args()
    
    run_live_recognition(simulate=args.simulate)
