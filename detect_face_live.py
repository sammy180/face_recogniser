import face_recognition
import cv2

print("[INFO] Initializing camera...")

# Start webcam
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

print("[INFO] Starting live webcam face detection...")
print("[INFO] Press 'q' to quit")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera")
        continue

    # Show original frame dimensions for debugging
    height, width = frame.shape[:2]
    
    # Resize frame for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    print(f"[DEBUG] Detected {len(face_locations)} faces in frame")

    # Draw rectangles
    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

    # Add status text to frame
    cv2.putText(frame, f"Camera: {width}x{height} | Faces: {len(face_locations)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Live Face Detection', frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting...")
        break

video_capture.release()
cv2.destroyAllWindows()
