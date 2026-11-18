import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# 1. Initialize the InsightFace model
# We use 'buffalo_l' which is the standard lightweight, highly accurate model
print("Loading AI Models... (This might take a moment to download the first time)")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 uses CPU. We can change this to GPU later on your laptop.

# 2. Define our universal relative paths
IMAGE_DIR = os.path.join(".", "dataset", "passport_photos")
DATABASE_DIR = os.path.join(".", "database")
OUTPUT_FILE = os.path.join(DATABASE_DIR, "student_embeddings.pkl")

# Ensure the database folder exists
os.makedirs(DATABASE_DIR, exist_ok=True)

# Dictionary to store { "student_name": [512_number_array] }
known_faces = {}

# 3. Loop through the passport photos
print(f"Scanning photos in: {IMAGE_DIR}")
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Get the student's name from the filename (e.g., "achala.jpg" -> "achala")
        student_name = os.path.splitext(filename)[0]
        
        image_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading {filename}. Skipping.")
            continue
            
        # 4. Extract the face data
        faces = app.get(img)
        
        # We only want photos with exactly ONE face in them to avoid database corruption
        if len(faces) == 1:
            # faces[0].embedding contains the 512-dimensional vector
            known_faces[student_name] = faces[0].embedding
            print(f"✅ Successfully enrolled: {student_name}")
        elif len(faces) == 0:
            print(f"❌ No face detected in {filename}.")
        else:
            print(f"⚠️ Multiple faces detected in {filename}. Skipping to maintain data purity.")

# 5. Save the dictionary to a local file
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(known_faces, f)

print(f"\nEnrollment Complete! Saved {len(known_faces)} student profiles to {OUTPUT_FILE}.")
