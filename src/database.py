import sqlite3
import pickle
import numpy as np
from datetime import datetime
from src.config import DB_PATH

def get_db_connection():
    """Establishes and returns a connection to the SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables if they do not exist."""
    schema_students = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_number TEXT UNIQUE NOT NULL,
        encoding BLOB,
        date_enrolled TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    schema_attendance = """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(schema_students)
        cursor.execute(schema_attendance)
        conn.commit()
    print(f"Database initialized successfully at: {DB_PATH}")

def add_student(name: str, roll_number: str, encoding: np.ndarray) -> bool:
    """
    Registers a new student with their face encoding.
    
    Args:
        name: Name of the student.
        roll_number: Unique roll number or ID.
        encoding: 128D numpy array representing the face encoding.
        
    Returns:
        True if successfully added, False otherwise.
    """
    # Serialize numpy array encoding to binary BLOB using pickle
    encoding_blob = pickle.dumps(encoding) if encoding is not None else None
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (name, roll_number, encoding) VALUES (?, ?, ?)",
                (name, roll_number, encoding_blob)
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError as e:
        print(f"Error adding student: Student with roll number {roll_number} already exists. Details: {e}")
        return False
    except Exception as e:
        print(f"Error adding student: {e}")
        return False

def get_all_students() -> list:
    """
    Retrieves all enrolled students with deserialized face encodings.
    
    Returns:
        List of dictionaries containing student records.
    """
    students_list = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, roll_number, encoding, date_enrolled FROM students")
            rows = cursor.fetchall()
            
            for row in rows:
                encoding = None
                if row['encoding']:
                    try:
                        encoding = pickle.loads(row['encoding'])
                    except Exception as e:
                        print(f"Error deserializing encoding for student ID {row['id']}: {e}")
                
                students_list.append({
                    "id": row["id"],
                    "name": row["name"],
                    "roll_number": row["roll_number"],
                    "encoding": encoding,
                    "date_enrolled": row["date_enrolled"]
                })
    except Exception as e:
        print(f"Error retrieving students: {e}")
    return students_list

def mark_attendance(student_id: int, date: str, time: str, status: str) -> bool:
    """
    Records attendance for a student.
    
    Args:
        student_id: Foreign key ID of the student.
        date: Date string (e.g., 'YYYY-MM-DD').
        time: Time string (e.g., 'HH:MM:SS').
        status: Attendance status (e.g., 'Present', 'Absent', 'Late').
        
    Returns:
        True if success, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
                (student_id, date, time, status)
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError as e:
        print(f"Error marking attendance: Student ID {student_id} does not exist. Details: {e}")
        return False
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False

def get_attendance_by_date(date: str) -> list:
    """
    Retrieves the attendance record for a specific date.
    
    Args:
        date: Date string (e.g., 'YYYY-MM-DD').
        
    Returns:
        List of dictionaries containing attendance records with student name and roll number.
    """
    attendance_list = []
    query = """
    SELECT a.id, a.student_id, s.name, s.roll_number, a.date, a.time, a.status 
    FROM attendance a 
    JOIN students s ON a.student_id = s.id 
    WHERE a.date = ?
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (date,))
            rows = cursor.fetchall()
            
            for row in rows:
                attendance_list.append({
                    "id": row["id"],
                    "student_id": row["student_id"],
                    "name": row["name"],
                    "roll_number": row["roll_number"],
                    "date": row["date"],
                    "time": row["time"],
                    "status": row["status"]
                })
    except Exception as e:
        print(f"Error retrieving attendance for date {date}: {e}")
    return attendance_list

def already_marked_today(student_id: int, date: str) -> bool:
    """
    Checks if a student has already marked attendance for a given date.
    
    Args:
        student_id: ID of the student.
        date: Date string (e.g., 'YYYY-MM-DD').
        
    Returns:
        True if already marked today, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
                (student_id, date)
            )
            return cursor.fetchone() is not None
    except Exception as e:
        print(f"Error checking daily attendance status: {e}")
        return False

if __name__ == "__main__":
    # Initialize DB when run directly
    init_db()
