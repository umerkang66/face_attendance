import streamlit as st
import pandas as pd
from datetime import datetime

# Import modules from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database import get_all_students, get_attendance_by_date
from src.config import RECOGNITION_THRESHOLD

st.set_page_config(page_title="Face Attendance System", page_icon="👤", layout="wide")

st.title("👤 Face Recognition Attendance Management System")
st.write("Welcome to the Attendance Dashboard. This is a computer vision course project.")

# Quick stats layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Status")
    st.info("System scaffolding set up. Ready to integrate Face Recognition pipeline.")
    st.metric(label="Recognition Distance Threshold", value=str(RECOGNITION_THRESHOLD))

with col2:
    st.subheader("Quick Statistics")
    try:
        students = get_all_students()
        st.metric(label="Total Enrolled Students", value=len(students))
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
