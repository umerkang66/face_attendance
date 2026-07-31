import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CAMERA_INDEX
from src.database import (
    init_db,
    get_all_students,
    get_attendance_by_date,
    mark_attendance,
    already_marked_today
)

# Run database initialization at startup
init_db()

# Page Configurations
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a Premium Look
st.markdown("""
<style>
    /* Global Background and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1f4068, #162447);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #e43f5a, #b12a3e);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(228, 63, 90, 0.4);
        color: white;
    }
    div.stButton > button:first-child:active {
        transform: translateY(1px);
    }
    
    /* Card Styles */
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div style="background: linear-gradient(135deg, #1f4068, #162447); padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
    <h1 style="margin: 0; font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; color: white;">👤 Face Recognition Attendance System</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 1.0rem; color: #a9bbd1;">Computer Vision Course Project Scaffolding & Control Panel</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Menu
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 10px;">
    <h3 style="margin-bottom: 0; font-size: 1.2rem; font-weight: 700; color: #162447;">Navigation Menu</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    ["Enroll Student", "Take Attendance", "Attendance Reports"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Configuration")
st.sidebar.write(f"**Camera Index:** `{CAMERA_INDEX}`")

# ----------------- PAGE 1: ENROLL STUDENT -----------------
if page == "Enroll Student":
    st.header("👤 Register New Student")
    st.write("Enroll a student by capturing 10 facial poses or running a mock enrollment simulation.")
    
    col_form, col_status = st.columns([1, 1])
    
    with col_form:
        name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
        roll_number = st.text_input("Roll Number / ID", placeholder="e.g. CS2026-08")
        simulate_enroll = st.checkbox("Simulate Enrollment (Use Mock Poses & Encoding)", value=False)
        
        start_btn = st.button("Start Enrollment Process", width='stretch')
        
    with col_status:
        st.subheader("Enrollment Status")
        status_box = st.empty()
        video_box = st.empty()
        
    if start_btn:
        if not name or not roll_number:
            st.error("Please fill in both Name and Roll Number fields.")
        else:
            from src.enroll import check_roll_number_exists
            if check_roll_number_exists(roll_number):
                st.error(f"Roll number '{roll_number}' is already registered in the system.")
            elif simulate_enroll:
                # Run simulated enrollment using database.py functions
                from src.enroll import enroll_student
                status_box.info("Running enrollment simulation...")
                success = enroll_student(name, roll_number, simulate=True)
                if success:
                    st.success(f"Success: Enrolled '{name}' (Roll: {roll_number}) with mock face profiles.")
                else:
                    st.error("Failed to simulate student registration.")
            else:
                # Real webcam capture loop rendered in Streamlit
                try:
                    import cv2
                    import numpy as np
                    import face_recognition
                    from src.enroll import INSTRUCTIONS, cleanup_folder
                    from src.config import DATASET_DIR
                    from src.database import add_student
                except ImportError as e:
                    st.error(f"Required libraries missing to open webcam: {e}. Please install them or use 'Simulate Enrollment'.")
                else:
                    # Sanitize folder path
                    sanitized_name = name.replace(" ", "_")
                    student_dir = DATASET_DIR / f"{sanitized_name}_{roll_number}"
                    student_dir.mkdir(parents=True, exist_ok=True)
                    
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        st.error("Failed to open camera. Confirm webcam is plugged in and index is correct.")
                    else:
                        encodings_list = []
                        step = 0
                        cooldown_duration = 2.0
                        cooldown_start = time.time()
                        warning_msg = None
                        
                        status_box.info("Camera online. Capture beginning...")
                        
                        # Show a stop button using streamlit keys
                        stop_clicked = st.button("Stop and Abort Capture")
                        
                        while step < len(INSTRUCTIONS):
                            if stop_clicked:
                                status_box.warning("Enrollment aborted by user.")
                                break
                                
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to read webcam stream.")
                                break
                                
                            display_frame = frame.copy()
                            h, w, _ = frame.shape
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            face_locations = face_recognition.face_locations(rgb_frame)
                            for top, right, bottom, left in face_locations:
                                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                
                            if len(face_locations) == 0:
                                warning_msg = "No face detected!"
                            elif len(face_locations) > 1:
                                warning_msg = "Multiple faces detected!"
                            else:
                                warning_msg = None
                                elapsed = time.time() - cooldown_start
                                if elapsed >= cooldown_duration:
                                    img_path = student_dir / f"img_{step + 1}.jpg"
                                    cv2.imwrite(str(img_path), frame)
                                    encs = face_recognition.face_encodings(rgb_frame, face_locations)
                                    if encs:
                                        encodings_list.append(encs[0])
                                        step += 1
                                        cooldown_start = time.time()
                                    else:
                                        warning_msg = "Encoding extraction failed."
                                        if img_path.exists():
                                            img_path.unlink()
                                            
                            # Bottom Overlay Bar
                            cv2.rectangle(display_frame, (0, h - 80), (w, h), (30, 30, 30), -1)
                            if step < len(INSTRUCTIONS):
                                cv2.putText(display_frame, f"Step {step+1}/10: {INSTRUCTIONS[step]}", 
                                            (15, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                                time_left = max(0.0, cooldown_duration - (time.time() - cooldown_start))
                                if warning_msg:
                                    cv2.putText(display_frame, f"STATUS: {warning_msg}", 
                                                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                else:
                                    cv2.putText(display_frame, f"Capturing pose in {time_left:.1f}s", 
                                                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                            
                            # Render Frame in Streamlit
                            video_box.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                            time.sleep(0.03)
                            
                        cap.release()
                        video_box.empty()
                        
                        if step == len(INSTRUCTIONS) and encodings_list:
                            avg_encoding = np.mean(encodings_list, axis=0)
                            success = add_student(name, roll_number, avg_encoding)
                            if success:
                                status_box.success(f"Successfully registered student '{name}' in the database.")
                            else:
                                status_box.error("Database integrity error. Roll number may already exist.")
                                cleanup_folder(student_dir)
                        else:
                            status_box.error("Enrollment failed or was incomplete. Cleaned up folders.")
                            cleanup_folder(student_dir)

# ----------------- PAGE 2: TAKE ATTENDANCE -----------------
elif page == "Take Attendance":
    st.header("📸 Attendance Recording Session")
    st.write("Track student attendance live via facial signatures.")
    
    # Session state to toggle capture loop
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
        
    simulate_session = st.checkbox("Simulate Attendance Session (Mocks Face Recognition)", value=False)
    
    col_actions, col_visual, col_table = st.columns([1, 2, 2])
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Define a persistent placeholder inside the table column to prevent redraw flickering
    with col_table:
        table_placeholder = st.empty()
    
    # Helper to update present student log table
    def show_marked_table():
        records = get_attendance_by_date(today_date)
        present_list = [r for r in records if r['status'] == 'Present']
        if present_list:
            df = pd.DataFrame(present_list)[['name', 'roll_number', 'time']]
            table_placeholder.dataframe(df, width='stretch')
        else:
            table_placeholder.info("No students marked present today yet.")
            
    with col_actions:
        st.subheader("Controls")
        if not st.session_state.session_active:
            start_session_btn = st.button("Start Session", width='stretch', type="primary")
            if start_session_btn:
                st.session_state.session_active = True
                st.rerun()
        else:
            end_session_btn = st.button("End Session & Finalize", width='stretch', type="secondary")
            if end_session_btn:
                st.session_state.session_active = False
                from src.attendance import mark_absent_students
                absent_count = mark_absent_students(today_date)
                st.success(f"Attendance finalized. {absent_count} student(s) marked Absent.")
                time.sleep(2)
                st.rerun()
                
        show_marked_table()
        
    with col_visual:
        st.subheader("Live Camera Stream")
        video_feed = st.empty()
        
    # Execute loop if session is active
    if st.session_state.session_active:
        all_students = get_all_students()
        if not all_students:
            st.error("No registered face records found in database! Please enroll a student first.")
            st.session_state.session_active = False
            st.rerun()
            
        if simulate_session:
            # Simulate a 10-second face recognition and marking cycle
            import random
            from PIL import Image, ImageDraw
            from src.attendance import write_log
            
            session_cache = {}
            
            for frame_idx in range(1, 11):
                # Choose random student to mock-detect
                student = random.choice(all_students)
                student_id = student['id']
                name = student['name']
                roll_number = student['roll_number']
                
                # Draw mock visual frame
                img = Image.new("RGB", (640, 480), color=(40, 40, 45))
                draw = ImageDraw.Draw(img)
                # Bounding box
                draw.rectangle([180, 100, 460, 380], outline=(0, 255, 0), width=4)
                draw.text((190, 350), f"SIMULATING: {name} (98%)", fill=(255, 255, 255))
                video_feed.image(img, width='stretch')
                
                # Handle Database logs
                now = time.time()
                last_check = session_cache.get(student_id, 0.0)
                if (now - last_check) > 3.0:
                    session_cache[student_id] = now
                    if not already_marked_today(student_id, today_date):
                        current_time = datetime.now().strftime("%H:%M:%S")
                        mark_attendance(student_id, today_date, current_time, "Present")
                        write_log(f"Streamlit App: Marked '{name}' (Roll: {roll_number}) as Present (Time: {current_time})")
                        
                show_marked_table()
                time.sleep(1.0)
                
            # Autoclose
            st.session_state.session_active = False
            from src.attendance import mark_absent_students
            absent_count = mark_absent_students(today_date)
            st.success(f"Simulation completed. Finalized: marked {absent_count} student(s) Absent.")
            time.sleep(2)
            st.rerun()
        else:
            # Real camera matching loop
            try:
                import cv2
                import numpy as np
                import face_recognition
                from src.recognizer import recognize_faces
                from src.attendance import write_log
            except ImportError as e:
                st.error(f"Required modules missing: {e}. Please run in 'Simulate' mode.")
                st.session_state.session_active = False
                st.rerun()
            else:
                roll_to_student_map = {s['roll_number']: s for s in all_students}
                cap = cv2.VideoCapture(CAMERA_INDEX)
                
                if not cap.isOpened():
                    st.error("Could not capture webcam. Confirm camera permissions.")
                    st.session_state.session_active = False
                    st.rerun()
                else:
                    session_cache = {}
                    newly_marked = {}
                    debounce_duration = 3.0
                    confirmation_duration = 2.0
                    
                    # Core loop
                    while st.session_state.session_active:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        display_frame = frame.copy()
                        h, w, _ = frame.shape
                        now_time = time.time()
                        
                        results = recognize_faces(frame)
                        
                        for face in results:
                            top, right, bottom, left = face['bbox']
                            name = face['name']
                            roll_number = face['roll_number']
                            confidence = face['confidence']
                            
                            is_newly_marked = False
                            
                            if name != "Unknown" and roll_number in roll_to_student_map:
                                student = roll_to_student_map[roll_number]
                                student_id = student['id']
                                
                                last_check = session_cache.get(student_id, 0.0)
                                if (now_time - last_check) > debounce_duration:
                                    session_cache[student_id] = now_time
                                    
                                    if not already_marked_today(student_id, today_date):
                                        current_time = datetime.now().strftime("%H:%M:%S")
                                        success = mark_attendance(student_id, today_date, current_time, "Present")
                                        if success:
                                            write_log(f"Streamlit App: Marked '{name}' (Roll: {roll_number}) as Present (Time: {current_time})")
                                            newly_marked[student_id] = now_time
                                            
                                marked_time = newly_marked.get(student_id, 0.0)
                                if (now_time - marked_time) < confirmation_duration:
                                    is_newly_marked = True
                                    
                            # Styling box color
                            if name == "Unknown":
                                color = (0, 0, 255)
                                label_text = "Unknown"
                            elif is_newly_marked:
                                color = (0, 255, 0)
                                label_text = f"Marked! {name} ({confidence*100:.0f}%)"
                            else:
                                color = (255, 255, 0)
                                label_text = f"Present: {name} ({confidence*100:.0f}%)"
                                
                            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(display_frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
                            cv2.putText(display_frame, label_text, (left + 6, bottom - 6), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                                        
                        # Update video element
                        video_feed.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                        show_marked_table()
                        time.sleep(0.03)
                        
                    cap.release()
                    video_feed.empty()

# ----------------- PAGE 3: ATTENDANCE REPORTS -----------------
elif page == "Attendance Reports":
    st.header("📊 Attendance Analytics & Reports")
    
    selected_date = st.date_input("Target Date", datetime.now().date())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    records = get_attendance_by_date(date_str)
    
    if records:
        df = pd.DataFrame(records)
        df_display = df[['name', 'roll_number', 'time', 'status']]
        
        # Dashboard metrics
        present_count = len(df_display[df_display['status'] == 'Present'])
        absent_count = len(df_display[df_display['status'] == 'Absent'])
        total_count = len(df_display)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="metric-card"><p style="margin:0; font-size: 0.9rem; color: #555;">Present</p><h2 style="margin:0; font-size:2.2rem; color:#28a745;">{present_count}</h2></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-card"><p style="margin:0; font-size: 0.9rem; color: #555;">Absent</p><h2 style="margin:0; font-size:2.2rem; color:#dc3545;">{absent_count}</h2></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-card"><p style="margin:0; font-size: 0.9rem; color: #555;">Total Class Strength</p><h2 style="margin:0; font-size:2.2rem; color:#1f4068;">{total_count}</h2></div>', unsafe_allow_html=True)
            
        st.write("### Today's Roll List")
        st.dataframe(df_display, width='stretch')
        
        # Download export data
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv_data,
            file_name=f"attendance_report_{date_str}.csv",
            mime="text/csv"
        )
        
        # Visual charts
        st.write("### Attendance Distribution Chart")
        chart_data = pd.DataFrame({
            'Status': ['Present', 'Absent'],
            'Count': [present_count, absent_count]
        })
        st.bar_chart(chart_data.set_index('Status'))
        
    else:
        st.info(f"No attendance logs were found in the database for the selected date: {date_str}.")
