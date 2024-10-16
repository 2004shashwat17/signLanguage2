import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr
import cv2


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

# Streamlit Sidebar Styling
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Sign Language Detection - Sameer Edlabadkar')
st.sidebar.subheader('- Parameter')

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# App Mode Selection
app_mode = st.sidebar.selectbox(
    'Choose the App mode', 
    ['About App', 'Sign Language to Text', 'Speech to Sign Language']
)

# About App Section
if app_mode == 'About App':
    st.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    st.markdown('This app uses **MediaPipe** for detecting Sign Language gestures, '
                '**SpeechRecognition** for converting voice to text, and Streamlit for the GUI.')

    st.video('https://youtu.be/NYAFEmte4og')
    st.markdown('''
        ### About Me  
        **Sameer Edlabadkar**  
        Working with **TensorFlow, MediaPipe, OpenCV, and ResNet50**.

        - [YouTube](https://www.youtube.com/@edlabadkarsameer/videos)
        - [LinkedIn](https://www.linkedin.com/in/sameer-edlabadkar-43b48b1a7/)
        - [GitHub](https://github.com/edlabadkarsameer)

        Feel free to reach out at **edlabadkarsameer@gmail.com**
    ''')

# Sign Language to Text Section
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"]
    )
    tffile = tempfile.NamedTemporaryFile(delete=False)

    # Initialize Video Capture
    if not video_file_buffer:
        vid = cv2.VideoCapture(0 if use_webcam else DEMO_VIDEO)
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    if not vid.isOpened():
        st.error("Error: Could not open video.")
        st.stop()

    codec = cv2.VideoWriter_fourcc(*'VP09')
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    while vid.isOpened():
        ret, img = vid.read()
        if not ret:
            st.warning("End of video stream or failed to read frame.")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        if record:
            out.write(img)

        frame_resized = image_resize(img, width=640)
        stframe.image(frame_resized, channels='BGR', use_column_width=True)

    vid.release()
    out.release()

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

# Speech to Sign Language Section
elif app_mode == 'Speech to Sign Language':
    st.title('Speech to Sign Language (Using Indian Sign Language)')

    r = sr.Recognizer()

    def display_images(text):
        img_dir = "images/"
        image_pos = st.empty()

        for char in text:
            if char.isalpha():
                img_path = os.path.join(img_dir, f"{char.lower()}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    image_pos.image(img, width=300)
                    time.sleep(2)
                    image_pos.empty()
            elif char == ' ':
                img_path = os.path.join(img_dir, "space.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    image_pos.image(img, width=300)
                    time.sleep(2)
                    image_pos.empty()

        time.sleep(2)
        image_pos.empty()

    if st.button("Start Talking"):
        with sr.Microphone() as source:
            st.write("Say something!")
            audio = r.listen(source, phrase_time_limit=5)

            try:
                text = r.recognize_google(audio)
                st.write(f"You said: {text}")
                display_images(text.lower())
            except sr.UnknownValueError:
                st.write("Sorry, I did not understand what you said.")
            except sr.RequestError as e:
                st.write(f"Could not request results; {e}")