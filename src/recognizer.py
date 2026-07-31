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
