# Face Recognition System

A real-time face recognition system using Python, OpenCV, and the face_recognition library. This system can train on known faces and recognize them in live camera feeds.

## Features

- **Real-time face detection and recognition** from webcam feed
- **Training system** to learn new faces from image datasets
- **Live camera feed** with labeled face boxes
- **Confidence scoring** for recognition accuracy
- **Multiple camera support** with automatic detection
- **Performance optimized** for smooth real-time operation
- **Pause/resume functionality** during live recognition
- **Screenshot capture** capability

## Project Structure

```
face_recogniser/
├── dataset/                    # Training images organized by person
│   └── Sammy/                 # Folder for each person (add more as needed)
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── Known/                     # Legacy folder (not used)
├── faces.pkl                  # Legacy encoding file
├── known_faces.pkl           # Main face encodings file
├── train_model.py            # Original training script
├── train_model_improved.py   # Improved training script (recommended)
├── recognise_face.py         # Basic face recognition script
├── detect_face_live.py       # Live face detection (no recognition)
├── face_recognition_stable.py # Stable face recognition (recommended)
├── test_camera.py            # Camera testing utility
└── README.md                 # This file
```

## Requirements

### Hardware
- Webcam or USB camera
- Computer with Python 3.7+

### Software Dependencies
- Python 3.7+
- conda (Anaconda/Miniconda)
- Required packages (installed via conda):
  - `dlib`
  - `face_recognition`
  - `opencv-python (cv2)`
  - `numpy`
  - `pickle`

## Installation

### Platform-Specific Instructions

#### Linux (Recommended Platform)
```bash
# 1. Clone or Download the Project
git clone <repository-url>
cd face_recogniser

# 2. Update system packages
sudo apt update
sudo apt install python3-pip python3-dev cmake

# 3. Set Up Conda Environment
conda create -n face_recognition python=3.12
conda activate face_recognition

# 4. Install required packages
conda install -c conda-forge dlib opencv numpy
pip install face_recognition

# 5. Verify Installation
python -c "import cv2, face_recognition, numpy; print('All packages installed successfully!')"
```

#### Windows
```bash
# Get a Linux machine.
# Seriously. Just get a Linux machine.
# Your life will be easier.
# Trust me on this one.
```

#### macOS
```bash
# Go home.
# This is a serious application for serious people.
# Come back when you have a real operating system.
```

### Alternative Installation (If You Insist on Using Inferior Systems)

If you absolutely must use Windows or macOS despite the clear guidance above:

#### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd face_recogniser
```

#### 2. Set Up Conda Environment
```bash
# Create and activate a Python virtual environment (recommended)
python -m venv venv
```
# Activate the enviroment :
```bash
source venv/bin/activate
```
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install dlib face_recognition opencv-python numpy
```

#### 3. Verify Installation
```bash
python -c "import cv2, face_recognition, numpy; print('All packages installed successfully!')"
```

## Usage

### Step 1: Prepare Training Data

1. Create a folder for each person in the `dataset/` directory:
   ```
   dataset/
   ├── Person1/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── photo3.jpg
   ├── Person2/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── ...
   ```

2. **Image Requirements:**
   - Clear, well-lit photos of faces
   - Multiple angles and expressions recommended
   - JPG, JPEG, or PNG format
   - Face should be clearly visible and not too small

### Step 2: Train the Model

**Option A: Improved Training (Recommended)**
```bash
python train_model_improved.py
```
- Choose option 1 for single averaged encoding per person
- This creates more robust and cleaner face representations

**Option B: Original Training**
```bash
python train_model.py
```
- Creates multiple encodings per person (one per image)

### Step 3: Test Camera

Before running face recognition, test your camera:
```bash
python test_camera.py
```
- Verifies camera access and functionality
- Press 'q' to quit, 's' to save screenshot

### Step 4: Run Face Recognition

**Option A: Stable Version (Recommended)**
```bash
python face_recognition_stable.py
```

**Option B: Basic Version**
```bash
python recognise_face.py
```

**Option C: Detection Only (No Recognition)**
```bash
python detect_face_live.py
```

## Controls

During live face recognition:
- **'q'** - Quit the application
- **'p'** - Pause/resume processing
- **'s'** - Save screenshot
- **ESC** - Alternative quit method

## How It Works

### Training Phase
1. **Image Processing**: Loads images from dataset folders
2. **Face Detection**: Finds faces in each image using dlib's HOG detector
3. **Feature Extraction**: Creates 128-dimensional face encodings
4. **Storage**: Saves encodings and names to `known_faces.pkl`

### Recognition Phase
1. **Camera Input**: Captures live video frames
2. **Performance Optimization**: Processes every nth frame for speed
3. **Face Detection**: Finds faces in current frame
4. **Face Recognition**: Compares detected faces with known encodings
5. **Confidence Scoring**: Calculates similarity scores
6. **Visual Output**: Draws labeled boxes around recognized faces

## Troubleshooting

### Camera Issues
```bash
# Test camera access
python test_camera.py

# Check available cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### Performance Issues
- Reduce camera resolution in the script
- Increase `process_every_n_frames` value
- Use fewer training images per person

### Recognition Issues
- Ensure good quality training images
- Add more diverse training images (different angles, lighting)
- Adjust tolerance values in recognition script
- Retrain with improved training script

### Package Installation Issues
```bash
# For dlib installation issues on Windows
conda install -c conda-forge dlib

# Alternative face_recognition installation
pip install cmake
pip install face_recognition
```

## Configuration

### Camera Settings
Modify these values in the recognition scripts:
```python
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Camera width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Camera height
video_capture.set(cv2.CAP_PROP_FPS, 15)            # Frame rate
```

### Recognition Settings
```python
process_every_n_frames = 2    # Process every nth frame
tolerance = 0.6               # Recognition sensitivity (lower = stricter)
```

## File Descriptions

- **`train_model_improved.py`**: Best training script with averaged encodings
- **`face_recognition_stable.py`**: Most reliable recognition script
- **`test_camera.py`**: Camera testing utility
- **`known_faces.pkl`**: Stores trained face encodings and names
- **`dataset/`**: Training images organized by person name

## Tips for Best Results

1. **Training Images**: Use 5-10 clear, diverse images per person
2. **Lighting**: Ensure good, even lighting in training photos
3. **Camera Position**: Position camera at eye level for best recognition
4. **Environment**: Consistent lighting during recognition improves accuracy
5. **Performance**: Close other applications for better camera performance

## Adding New People

1. Create a new folder in `dataset/` with the person's name
2. Add 5-10 good quality photos of their face
3. Run the training script again
4. The system will automatically include the new person

## Technical Details

- **Face Detection**: dlib's HOG-based detector
- **Face Recognition**: 128-dimensional face encodings
- **Similarity Metric**: Euclidean distance
- **Performance**: ~15-30 FPS depending on hardware
- **Accuracy**: >95% with good training data

## License

This project is for educational and personal use. Please respect privacy and obtain consent before using face recognition technology on others.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Test with the camera testing script first
4. Ensure training data is properly formatted
