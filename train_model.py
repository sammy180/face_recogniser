import face_recognition
import os
import pickle

DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'known_faces.pkl'

known_encodings = []
known_names = []

print("[INFO] Starting training...")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    
    if not os.path.isdir(person_dir):
        continue
    
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        print(f"[INFO] Processing {image_path}")

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"[WARNING] No face found in {image_name}")

# Save encodings
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump((known_encodings, known_names), f)

print("[INFO] Training completed. Encodings saved.")
