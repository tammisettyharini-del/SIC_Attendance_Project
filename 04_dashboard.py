import streamlit as st
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- Universal Paths ---
DB_PATH = os.path.join(".", "database", "student_embeddings.pkl")
YOLO_MODEL_PATH = os.path.join(".", "database", "id_card_model", "weights", "best.pt")

def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# --- Dashboard UI Configuration ---
st.set_page_config(page_title="SIC AI Attendance", layout="wide")
st.title("🛡️ SIC Hackathon: Edge AI & Privacy-First Attendance")
st.markdown("Processing CCTV footage entirely offline. No cloud APIs. No privacy leaks.")

# --- Load Models Once Using Cache ---
# @st.cache_resource ensures we don't reload heavy AI models every time a button is clicked
@st.cache_resource
def load_ai_models():
    if not os.path.exists(YOLO_MODEL_PATH) or not os.path.exists(DB_PATH):
        return None, None, None
        
    with open(DB_PATH, 'rb') as f:
        known_faces = pickle.load(f)
        
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    id_detector = YOLO(YOLO_MODEL_PATH)
    return known_faces, face_app, id_detector

known_faces, face_app, id_detector = load_ai_models()

if known_faces is None:
    st.error("⚠️ Missing Database or YOLO model. Please run Phase 1 and Phase 2 first!")
    st.stop()

# --- Layout: 2 Columns ---
col1, col2 = st.columns([2, 1]) # Make the video column wider

with col1:
    st.markdown("### 🎥 Live CCTV Feed")
    run = st.checkbox('Start System') # A button to turn the camera on/off
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("### 📋 Live Attendance Log")
    log_placeholder = st.empty() # Placeholder for our data table
    
attendance_log = [] # List to keep track of who we saw

# --- Main Video Loop ---
if run:
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read webcam.")
            break
            
        # 1. Detect ID Cards
        id_results = id_detector(frame, conf=0.5, verbose=False)
        id_detected = len(id_results[0].boxes) > 0 
        
        # 2. Detect Faces
        faces = face_app.get(frame)
        
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            best_match_name = "Unknown"
            highest_sim = 0.0

            # 3. Match Face
            for name, db_embedding in known_faces.items():
                sim = compute_similarity(face.embedding, db_embedding)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_name = name

            # 4. Apply Business Logic & Drawing
            if highest_sim > 0.45:
                # If they have an ID, draw green. If no ID, draw yellow.
                status = "Present & ID Verified" if id_detected else "Present (NO ID)"
                bgr_color = (0, 255, 0) if id_detected else (0, 255, 255)
                
                # Add to attendance log if we haven't logged them yet
                if best_match_name not in [log["USN"] for log in attendance_log]:
                    attendance_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "USN": best_match_name,
                        "Status": status
                    })
            else:
                status = "Unknown Person"
                bgr_color = (0, 0, 255) # Red

            # Draw boxes on the video frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.putText(frame, f"{best_match_name}: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
            
        # OpenCV uses BGR colors, but Streamlit expects RGB. We must convert it.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        # Update the live table on the side
        if attendance_log:
            df = pd.DataFrame(attendance_log)
            # Reorder columns for clean display
            df = df[["Time", "USN", "Status"]]
            log_placeholder.dataframe(df, use_container_width=True, hide_index=True)
            
    cap.release()
