import face_recognition
import os
import pickle
import numpy as np

DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'known_faces.pkl'

def train_faces_single_encoding_per_person():
    """Create a single averaged encoding per person"""
    print("[INFO] Training with single encoding per person...")
    
    known_encodings = []
    known_names = []
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"[INFO] Processing person: {person_name}")
        person_encodings = []
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            print(f"  - Processing {image_name}")
            
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    person_encodings.append(encodings[0])
                    print(f"    ✓ Face encoding extracted")
                else:
                    print(f"    ✗ No face found in {image_name}")
            except Exception as e:
                print(f"    ✗ Error processing {image_name}: {e}")
        
        if person_encodings:
            # Average all encodings for this person
            average_encoding = np.mean(person_encodings, axis=0)
            known_encodings.append(average_encoding)
            known_names.append(person_name)
            print(f"  → Created averaged encoding from {len(person_encodings)} images")
        else:
            print(f"  → No valid face encodings found for {person_name}")
    
    return known_encodings, known_names

def train_faces_multiple_encodings_per_person():
    """Keep multiple encodings per person (original method)"""
    print("[INFO] Training with multiple encodings per person...")
    
    known_encodings = []
    known_names = []
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"[INFO] Processing person: {person_name}")
        person_count = 0
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            print(f"  - Processing {image_name}")
            
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    person_count += 1
                    print(f"    ✓ Face encoding extracted")
                else:
                    print(f"    ✗ No face found in {image_name}")
            except Exception as e:
                print(f"    ✗ Error processing {image_name}: {e}")
        
        print(f"  → Created {person_count} encodings for {person_name}")
    
    return known_encodings, known_names

def main():
    print("Face Recognition Training Script")
    print("=" * 40)
    print("Choose training method:")
    print("1. Single averaged encoding per person (recommended)")
    print("2. Multiple encodings per person (original)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    if choice == '1':
        known_encodings, known_names = train_faces_single_encoding_per_person()
    else:
        known_encodings, known_names = train_faces_multiple_encodings_per_person()
    
    if not known_encodings:
        print("[ERROR] No face encodings were created!")
        return
    
    # Save encodings
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"- Total encodings: {len(known_encodings)}")
    print(f"- Unique people: {len(set(known_names))}")
    print(f"- People: {list(set(known_names))}")
    print(f"- Encodings saved to: {ENCODINGS_FILE}")

if __name__ == "__main__":
    main()
