1️⃣ Streamlit

What it is: An open-source Python framework for quickly building and sharing data apps and ML demos through a simple web interface.

Why you used it: To create the web-based application for your sign language game without writing complex frontend code.

Benefit: Lets you display your webcam feed, model outputs, and instructions in a user-friendly way.

2️⃣ MediaPipe

What it is: A Google framework for building perception pipelines (face detection, hand tracking, pose estimation, etc.).

Why you used it: For real-time hand tracking and extracting hand landmarks (like fingertip coordinates) from the webcam feed, which are essential for detecting specific sign language gestures.

Benefit: Very fast, works in real-time even without a GPU, and gives high-accuracy hand keypoints.

3️⃣ OpenCV-Python (cv2)

What it is: The Python binding for OpenCV, an open-source computer vision library.

Why you used it:

Capturing frames from the webcam

Drawing bounding boxes/landmarks over hands

Processing images before passing them to the model

Benefit: Flexible and efficient for image processing and camera handling.

4️⃣ NumPy

What it is: The most popular Python library for numerical computing and array operations.

Why you used it:

Handling image data as arrays

Performing mathematical calculations for gesture recognition

Reshaping/normalizing data before feeding it into models

Benefit: Very fast and works seamlessly with OpenCV and ML libraries.

5️⃣ SpeechRecognition

What it is: A Python library for recognizing speech from audio using various APIs (like Google Speech API).

Why you used it: To allow the system to provide audio feedback or convert spoken results to text if needed (e.g., confirming gesture detection with voice).

Benefit: Makes the application interactive and accessible for users with multiple input/output modes.

6️⃣ Pillow

What it is: A Python Imaging Library (PIL) fork for opening, manipulating, and saving image files.

Why you used it:

Handling and displaying images in Streamlit

Possibly for overlaying gesture icons, instructions, or processed frames

Benefit: Easy image handling and compatible with both OpenCV and Streamlit. 


For my sign language game, I used Streamlit for the web interface, MediaPipe for real-time hand landmark detection,
and OpenCV-Python for image capture and processing. NumPy handled numerical operations, SpeechRecognition enabled voice interaction,
and Pillow was used for image manipulation. Together, these tools allowed me to create an interactive sign language recognition app 
that works in real-time through a browser.
