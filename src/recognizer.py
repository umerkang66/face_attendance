import sys
from pathlib import Path
import numpy as np

# Import modules from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RECOGNITION_THRESHOLD
from src.database import get_all_students

# Module-level cache for known encodings to avoid repeated database queries
_KNOWN_ENCODINGS = []
_KNOWN_METADATA = []  # List of dicts: {'name': name, 'roll_number': roll_number}

def load_known_encodings(force_reload: bool = False) -> tuple:
    """
    Loads all registered student encodings and names from the database into memory.
    Caches the results globally for fast comparisons in video loops.
    
    Args:
        force_reload: If True, bypasses cache and re-queries database.
        
    Returns:
        A tuple of (known_encodings, known_metadata)
    """
    global _KNOWN_ENCODINGS, _KNOWN_METADATA
    
    if not force_reload and _KNOWN_ENCODINGS:
        return _KNOWN_ENCODINGS, _KNOWN_METADATA
        
    _KNOWN_ENCODINGS = []
    _KNOWN_METADATA = []
    
    students = get_all_students()
    for student in students:
        if student['encoding'] is not None:
            _KNOWN_ENCODINGS.append(student['encoding'])
            _KNOWN_METADATA.append({
                'name': student['name'],
                'roll_number': student['roll_number']
            })
            
    print(f"[RECOGNIZER] Loaded {len(_KNOWN_ENCODINGS)} face signatures from database.")
    return _KNOWN_ENCODINGS, _KNOWN_METADATA

def recognize_faces(frame, threshold: float = RECOGNITION_THRESHOLD) -> list:
    """
    Detects faces in the given BGR frame and matches them against known student signatures.
    
    Args:
        frame: BGR image frame (NumPy array) from OpenCV.
        threshold: Distance threshold for matching. Lower is stricter.
        
    Returns:
        List of dicts: [{name, roll_number, confidence, bbox}]
        Each bbox is a tuple: (top, right, bottom, left)
    """
    # Lazy imports of face_recognition and cv2 to support import/scaffolding checks
    try:
        import face_recognition
        import cv2
    except ImportError as e:
        print(f"[ERROR] Required computer vision libraries missing for recognize_faces: {e}")
        return []
        
    # Ensure known signatures are loaded
    known_encodings, known_metadata = load_known_encodings()
    
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect all face locations in the frame
    # NOTE: We use "hog" (Histogram of Oriented Gradients) because it is fast and runs in real-time on CPU.
    # ALTERNATIVE: Use model="cnn" for a deep CNN face detector. It is significantly more accurate,
    # particularly for non-frontal faces or varying angles, but is too slow for real-time video on standard CPUs (ideal for GPU).
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    # Extract 128D face encodings for detected face positions
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    for bbox, face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        roll_number = None
        min_distance = 1.0
        
        if known_encodings:
            # Calculate Euclidean distance between the current face and all known faces
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Find the best matching index
            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]
            
            # Verify if the match meets our confidence/distance threshold
            if best_distance <= threshold:
                name = known_metadata[best_match_idx]['name']
                roll_number = known_metadata[best_match_idx]['roll_number']
                min_distance = best_distance
                
        # Confidence score metric
        # dlib face recognition embeddings output face distance where 0.6 is the standard threshold.
        # We model a normalized confidence score based on the distance.
        if min_distance <= threshold:
            # Scales from 100% confidence (0.0 distance) to 50% confidence (at threshold)
            confidence = 1.0 - (min_distance / (threshold * 2.0))
        else:
            # Degrades confidence quickly beyond the matching threshold
            confidence = max(0.0, 1.0 - min_distance)
            
        results.append({
            'name': name,
            'roll_number': roll_number,
            'confidence': float(confidence),
            'bbox': bbox  # (top, right, bottom, left)
        })
        
    return results

def draw_face_label(frame, bbox, label_text, color=(0, 255, 0), top_badge=None):
    """
    Draws a highly visible face bounding box and text label overlay.
    Dynamically sizes background badge to fit text, enforces high-contrast colors,
    and prevents text truncation near video edges.
    """
    import cv2
    
    top, right, bottom, left = bbox
    h, w, _ = frame.shape
    
    # 1. Main face bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # 2. Modern corner accents
    corner_len = min(15, (right - left) // 4, (bottom - top) // 4)
    if corner_len > 3:
        cv2.line(frame, (left, top), (left + corner_len, top), color, 3)
        cv2.line(frame, (left, top), (left, top + corner_len), color, 3)
        cv2.line(frame, (right, top), (right - corner_len, top), color, 3)
        cv2.line(frame, (right, top), (right, top + corner_len), color, 3)
        cv2.line(frame, (left, bottom), (left + corner_len, bottom), color, 3)
        cv2.line(frame, (left, bottom), (left, bottom - corner_len), color, 3)
        cv2.line(frame, (right, bottom), (right - corner_len, bottom), color, 3)
        cv2.line(frame, (right, bottom), (right, bottom - corner_len), color, 3)

    # 3. Measure text dimensions dynamically
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    pad_x = 8
    pad_y = 6
    box_w = max(right - left, text_w + 2 * pad_x)
    box_h = text_h + baseline + 2 * pad_y
    
    # Clamp badge horizontally within frame dimensions
    bg_left = max(0, min(left, w - box_w))
    bg_right = min(w, bg_left + box_w)
    
    # Clamp badge vertically (draw below face, or above if bottom edge overflows frame)
    if bottom + box_h <= h:
        bg_top = bottom
        bg_bottom = bottom + box_h
        text_y = bg_top + text_h + pad_y
    else:
        bg_top = max(0, top - box_h)
        bg_bottom = top
        text_y = bg_top + text_h + pad_y
        
    text_x = bg_left + pad_x
    
    # 4. Fill background badge
    cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), color, cv2.FILLED)
    cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), 1)
    
    # 5. Determine high-contrast text color based on background luminance
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 120 else (255, 255, 255)
    
    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # 6. Draw optional top badge (e.g. Roll number)
    if top_badge:
        (tb_w, tb_h), tb_base = cv2.getTextSize(top_badge, font, 0.45, 1)
        tb_box_w = tb_w + 12
        tb_box_h = tb_h + tb_base + 6
        
        tb_left = left
        tb_right = left + tb_box_w
        tb_top = max(0, top - tb_box_h)
        tb_bottom = tb_top + tb_box_h
        
        cv2.rectangle(frame, (tb_left, tb_top), (tb_right, tb_bottom), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (tb_left, tb_top), (tb_right, tb_bottom), color, 1)
        cv2.putText(frame, top_badge, (tb_left + 6, tb_top + tb_h + 3), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

"""
====================================================================================================
PROJECT REPORT DOCUMENTATION: UNDERSTANDING THE DISTANCE THRESHOLD
====================================================================================================
In dlib-based facial recognition, the face encoder maps a face image into a 128-dimensional vector space 
such that distance between vectors corresponds to facial similarity. The distance threshold determines 
whether a query face is classified as a 'match' or 'mismatch' relative to database records.

This threshold directly controls the trade-off between the following performance metrics:

1. False-Accept Rate (FAR) / False Match Rate:
   - The probability that the system incorrectly identifies an unregistered person (or a different registered
     person) as a target student.
   - Affects system security and attendance integrity.

2. False-Reject Rate (FRR) / False Non-Match Rate:
   - The probability that the system fails to recognize a registered student, classifying them as "Unknown".
   - Affects system usability and student convenience.

Trade-off Dynamics:
----------------------------------------------------------------------------------------------------
  Threshold Level  |  False-Accept Rate (FAR)  |  False-Reject Rate (FRR)  |  Characteristics
----------------------------------------------------------------------------------------------------
  Low (e.g. 0.4)   |        Very Low           |          High             |  Extremely strict. Very secure.
                   |  Strangers never get      |  Registered students get  |  High rejection rate under 
                   |  recognized.              |  rejected.                |  improper lighting or angles.
----------------------------------------------------------------------------------------------------
  Medium (0.6)     |        Balanced           |        Balanced           |  Dlib's default. Optimal 
                   |  Highly accurate match    |  Good tolerance for minor |  operating point for 
                   |  guarantees.              |  profile changes.         |  general verification.
----------------------------------------------------------------------------------------------------
  High (e.g. 0.8)  |        High               |          Low              |  Highly lenient. Strangers 
                   |  Strangers easily get     |  Students always pass,    |  often register false matches.
                   |  recognized as students.  |  even with severe blur.   |  Not suitable for attendance.
----------------------------------------------------------------------------------------------------
"""
