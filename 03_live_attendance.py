import cv2
import pickle
import numpy as np
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- Universal Paths ---
DB_PATH = os.path.join(".", "database", "student_embeddings.pkl")
YOLO_MODEL_PATH = os.path.join(".", "database", "id_card_model", "weights", "best.pt")

# Helper function for Vector Math (Cosine Similarity)
def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def main():
    print("Initializing Edge AI Pipeline...")

    # 1. Check if YOLO model exists (Prevents crashes if you haven't trained yet)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"\n[ERROR] YOLO model not found at {YOLO_MODEL_PATH}")
        print("You must run 02_train_id_detector.py on your laptop first!\n")
        return

    # 2. Load the Student Database
    if not os.path.exists(DB_PATH):
        print("[ERROR] Database not found! Run Phase 1 first.")
        return
        
    with open(DB_PATH, 'rb') as f:
        known_faces = pickle.load(f)
    print(f"Loaded {len(known_faces)} student profiles.")

    # 3. Load AI Models
    print("Loading Face Analysis & YOLO models...")
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    id_detector = YOLO(YOLO_MODEL_PATH)

    # 4. Open Video Stream (0 is webcam, or replace with 'cctv_sample.mp4')
    cap = cv2.VideoCapture(0) 

    print("\n--- Live Attendance Active (Press 'q' to quit) ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Detect ID Cards using YOLO
        # conf=0.5 means it only counts if it's 50% sure it's an ID
        id_results = id_detector(frame, conf=0.5, verbose=False)
        id_detected = len(id_results[0].boxes) > 0 

        # B. Detect Faces using InsightFace
        faces = face_app.get(frame)

        for face in faces:
            # Get face bounding box coordinates
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            
            best_match_name = "Unknown"
            highest_sim = 0.0

            # C. Compare live face to database
            for name, db_embedding in known_faces.items():
                sim = compute_similarity(face.embedding, db_embedding)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_name = name

            # D. Logic & Display (Threshold 0.45 is standard for ArcFace)
            if highest_sim > 0.45:
                if id_detected:
                    label = f"{best_match_name} (ID: OK)"
                    color = (0, 255, 0) # Green
                else:
                    label = f"{best_match_name} (NO ID)"
                    color = (0, 255, 255) # Yellow
            else:
                label = "Unknown Person"
                color = (0, 0, 255) # Red

            # Draw bounding box and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show the live feed
        cv2.imshow("SIC Edge AI Attendance", frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
