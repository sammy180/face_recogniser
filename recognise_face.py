import face_recognition
import cv2
import pickle
import os

# Load known face encodings
ENCODINGS_FILE = 'known_faces.pkl'
if not os.path.exists(ENCODINGS_FILE):
    print(f"[ERROR] {ENCODINGS_FILE} not found! Please run train_model.py first.")
    exit(1)

try:
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)
    print(f"[INFO] Loaded {len(known_names)} known faces: {known_names}")
except Exception as e:
    print(f"[ERROR] Failed to load known faces: {e}")
    exit(1)

# Start webcam
print("[INFO] Initializing camera...")
video_capture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_capture.isOpened():
    print("[ERROR] Could not open camera. Trying different camera indices...")
    for i in range(1, 4):
        video_capture = cv2.VideoCapture(i)
        if video_capture.isOpened():
            print(f"[INFO] Camera found at index {i}")
            break
    else:
        print("[ERROR] No camera found!")
        exit(1)

# Set camera properties for better performance
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] Starting webcam for real-time recognition...")
print("[INFO] Press 'q' to quit")

frame_count = 0
process_every_n_frames = 3  # Process every 3rd frame for better performance
face_locations = []
face_encodings = []
face_names = []

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera")
            continue

        # Show original frame dimensions for debugging (only once)
        if frame_count == 0:
            height, width = frame.shape[:2]
            print(f"[INFO] Camera resolution: {width}x{height}")
        
        frame_count += 1

        # Only process every nth frame for face recognition (for performance)
        if frame_count % process_every_n_frames == 0:
            # Resize and convert to RGB
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Detect face locations and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                confidence = 0.0

                if True in matches:
                    # Find the best match
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        print(f"[INFO] Recognized: {name} (confidence: {confidence:.2f})")

                face_names.append(f"{name} ({confidence:.2f})")

        # Draw the results on every frame (even if we didn't process it)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale face box coordinates back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Choose color based on recognition
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            
            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name label with background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add status information
        status_text = f"Frame: {frame_count} | Known faces: {len(known_names)} | Detected: {len(face_locations)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow('Face Recognition - Live Feed', frame)

        # Handle key presses with minimal delay
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
        elif key == ord(' '):  # Spacebar to pause/resume processing
            print("[INFO] Processing paused. Press any key to continue...")
            cv2.waitKey(0)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")
finally:
    print("[INFO] Cleaning up...")

video_capture.release()
cv2.destroyAllWindows()
