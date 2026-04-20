import cv2, mediapipe as mp, numpy as np, time, os, tempfile
from PIL import Image
import streamlit as st
from collections import deque
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os

# Paths
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_APP_DIR = _os.path.dirname(_BASE_DIR)

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
confidence_threshold = 0.5


# ✅ Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1, 1
            )
        )
    )


# ✅ REQUIRED FUNCTION (fixes ImportError)
def glute_bridge_detection():
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Key points
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                angle = calculate_angle(shoulder, hip, knee)

                # Glute bridge logic
                if angle < 120:
                    stage = "down"

                if angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1

                # Show angle
                cv2.putText(image, str(int(angle)),
                            tuple(np.multiply(hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except:
                pass

            # Draw pose
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # UI
            cv2.rectangle(image, (0, 0), (260, 80), (0, 0, 0), -1)

            cv2.putText(image, 'GLUTE BRIDGE', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(image, f'Reps: {counter}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Glute Bridge Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()