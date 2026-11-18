
import streamlit as st
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# =====================================================
# 1. PROJECT PATHS
# =====================================================

DB_PATH = os.path.join(
    ".", "database", "student_embeddings.pkl"
)

YOLO_MODEL_PATH = os.path.join(
    ".", "database", "id_card_model", "weights", "best.pt"
)


# =====================================================
# 2. FACE SIMILARITY
# =====================================================

def compute_similarity(emb1, emb2):
    denominator = (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(emb1, emb2) / denominator)


# =====================================================
# 3. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SIC AI Attendance",
    layout="wide"
)

st.title("🛡️ SIC Hackathon: Edge AI & Privacy-First Attendance")

st.markdown(
    "Processing CCTV footage locally using AI-based "
    "face recognition and ID-card detection."
)


# =====================================================
# 4. LOAD AI MODELS
# =====================================================

@st.cache_resource
def load_ai_models():

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Student database not found: {DB_PATH}"
        )

    with open(DB_PATH, "rb") as file:
        known_faces = pickle.load(file)

    # CPU mode because this system currently uses CPU
    face_app = FaceAnalysis(name="buffalo_l")

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    id_detector = YOLO(YOLO_MODEL_PATH)

    return known_faces, face_app, id_detector


try:
    known_faces, face_app, id_detector = load_ai_models()

except Exception as error:
    st.error(f"AI Model Loading Error: {error}")
    st.stop()


# =====================================================
# 5. DASHBOARD LAYOUT
# =====================================================

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("### 🎥 Live Camera Feed")

    run = st.checkbox("Start System")

    FRAME_WINDOW = st.empty()


with col2:

    st.markdown("### 📋 Live Attendance Log")

    log_placeholder = st.empty()


attendance_log = []


# =====================================================
# 6. LIVE CAMERA PROCESSING
# =====================================================

if run:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error("Unable to open webcam.")

    else:

        try:

            while run:

                ret, frame = cap.read()

                if not ret:

                    st.error("Failed to read webcam.")
                    break


                # -------------------------------------
                # 6A. ID CARD DETECTION USING YOLO
                # -------------------------------------

                id_results = id_detector(
                    frame,
                    conf=0.20,
                    verbose=False
                )

                id_detected = False

                for detection in id_results[0].boxes:

                    class_id = int(detection.cls[0])

                    confidence = float(
                        detection.conf[0]
                    )

                    class_name = id_detector.names[class_id]

                    # Class 0 = ID Card
                    if class_id == 0:
                        id_detected = True

                    x1, y1, x2, y2 = map(
                        int,
                        detection.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{class_name}: {confidence:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )


                # -------------------------------------
                # 6B. FACE DETECTION
                # -------------------------------------

                faces = face_app.get(frame)

                for face in faces:

                    box = face.bbox.astype(int)

                    x1, y1, x2, y2 = box

                    best_match_name = "Unknown"

                    highest_sim = 0.0


                    # ---------------------------------
                    # 6C. FACE RECOGNITION
                    # ---------------------------------

                    for name, db_embedding in known_faces.items():

                        sim = compute_similarity(
                            face.embedding,
                            db_embedding
                        )

                        if sim > highest_sim:

                            highest_sim = sim

                            best_match_name = name


                    # ---------------------------------
                    # 6D. ATTENDANCE STATUS
                    # ---------------------------------

                    if highest_sim > 0.45:

                        if id_detected:

                            status = "Present & ID Verified"

                            bgr_color = (0, 255, 0)

                        else:

                            status = "Present (NO ID)"

                            bgr_color = (0, 255, 255)


                        # Check whether student already exists
                        existing_entry = next(
                            (
                                log for log in attendance_log
                                if log["USN"] == best_match_name
                            ),
                            None
                        )

                        if existing_entry is None:

                            attendance_log.append({

                                "Time": datetime.now().strftime(
                                    "%H:%M:%S"
                                ),

                                "USN": best_match_name,

                                "Status": status
                            })

                        elif id_detected:

                            existing_entry["Status"] = status


                    else:

                        status = "Unknown Person"

                        bgr_color = (0, 0, 255)


                    # ---------------------------------
                    # 6E. DRAW FACE BOUNDING BOX
                    # ---------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        bgr_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{best_match_name}: {status}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        bgr_color,
                        2
                    )


                # -------------------------------------
                # 6F. DISPLAY VIDEO
                # -------------------------------------

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                FRAME_WINDOW.image(frame_rgb)


                # -------------------------------------
                # 6G. UPDATE ATTENDANCE TABLE
                # -------------------------------------

                if attendance_log:

                    df = pd.DataFrame(attendance_log)

                    df = df[
                        ["Time", "USN", "Status"]
                    ]

                    log_placeholder.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

        finally:

            cap.release()

else:

    st.info("Click 'Start System' to activate the camera.")
