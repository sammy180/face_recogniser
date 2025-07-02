import face_recognition
import cv2
import pickle
import os
import time
import numpy as np

def load_known_faces():
    """Load known face encodings from pickle file"""
    encodings_file = 'known_faces.pkl'
    
    if not os.path.exists(encodings_file):
        print(f"[ERROR] {encodings_file} not found!")
        print("[INFO] Available files:", os.listdir('.'))
        return None, None
    
    try:
        with open(encodings_file, 'rb') as f:
            known_encodings, known_names = pickle.load(f)
        print(f"[INFO] Loaded {len(known_names)} known faces: {known_names}")
        return known_encodings, known_names
    except Exception as e:
        print(f"[ERROR] Failed to load known faces: {e}")
        return None, None

def initialize_camera():
    """Initialize camera with error handling"""
    print("[INFO] Initializing camera...")
    
    # Try different camera indices
    for camera_index in range(5):
        video_capture = cv2.VideoCapture(camera_index)
        if video_capture.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = video_capture.read()
            if ret:
                print(f"[INFO] Camera successfully opened at index {camera_index}")
                
                # Set camera properties for optimal performance
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                video_capture.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
                
                return video_capture
            video_capture.release()
    
    print("[ERROR] No working camera found!")
    return None

def main():
    """Main face recognition loop"""
    # Load known faces
    known_encodings, known_names = load_known_faces()
    if known_encodings is None:
        return
    
    # Initialize camera
    video_capture = initialize_camera()
    if video_capture is None:
        return
    
    print("[INFO] Starting face recognition...")
    print("[INFO] Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 's' to save screenshot")
    
    # Performance settings
    process_every_n_frames = 2  # Process every 2nd frame
    frame_count = 0
    paused = False
    
    # Initialize variables for face tracking
    face_locations = []
    face_names = []
    
    try:
        while True:
            if not paused:
                ret, frame = video_capture.read()
                if not ret:
                    print("[WARNING] Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process face recognition every nth frame for performance
                if frame_count % process_every_n_frames == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Find faces
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
                        face_names = []
                        for face_encoding in face_encodings:
                            # Compare with known faces
                            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                            name = "Unknown"
                            confidence = 0.0
                            
                            if True in matches:
                                # Get face distances for confidence score
                                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index]:
                                    name = known_names[best_match_index]
                                    confidence = 1 - face_distances[best_match_index]
                                    print(f"[RECOGNITION] {name} (confidence: {confidence:.2f})")
                            
                            face_names.append((name, confidence))
                    else:
                        face_names = []
                
                # Draw results on frame
                for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
                    # Scale coordinates back to original frame size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Choose color based on recognition
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green for known faces
                        display_name = f"{name} ({confidence:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown faces
                        display_name = "Unknown"
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # Draw name
                    cv2.putText(frame, display_name, (left + 6, bottom - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add status information
                status_y = 30
                cv2.putText(frame, f"Frame: {frame_count} | Faces: {len(face_locations)}", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Known: {len(known_names)} people", 
                           (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Press 'q'=quit, 'p'=pause, 's'=screenshot", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            else:
                # Show paused message
                height, width = frame.shape[:2] if 'frame' in locals() else (480, 640)
                pause_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(pause_frame, "PAUSED - Press 'p' to resume", 
                           (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                frame = pause_frame
            
            # Display the frame
            cv2.imshow('Face Recognition - Live Feed', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and not paused:
                screenshot_name = f"recognition_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"[INFO] Screenshot saved: {screenshot_name}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        video_capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Done!")

if __name__ == "__main__":
    main()
