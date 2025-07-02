import cv2
import time

print("[INFO] Testing camera access...")

# Try to open camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Cannot access camera index 0. Trying other indices...")
    for i in range(1, 5):
        video_capture = cv2.VideoCapture(i)
        if video_capture.isOpened():
            print(f"[INFO] Camera found at index {i}")
            break
    else:
        print("[ERROR] No camera found!")
        exit(1)

# Set camera properties
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Camera initialized successfully!")
print("[INFO] Press 'q' to quit, 's' to save screenshot")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[INFO] FPS: {fps:.1f}")
        
        # Add frame info
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Frame: {frame_count} | Resolution: {width}x{height}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Camera Test', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"[INFO] Screenshot saved as {screenshot_name}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    print("[INFO] Cleaning up...")
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Done!")
