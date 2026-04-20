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


# ✅ REQUIRED FUNCTION (this fixes your import error)
def arnold_press_detection():
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Right arm
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Arnold press logic
                if angle < 90:
                    stage = "down"
                if angle > 150 and stage == "down":
                    stage = "up"
                    counter += 1

                # Show angle
                cv2.putText(image, str(int(angle)),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except:
                pass

            # Draw pose
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # UI box
            cv2.rectangle(image, (0, 0), (250, 80), (0, 0, 0), -1)

            cv2.putText(image, 'REPS', (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, str(counter), (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            cv2.putText(image, 'STAGE', (120, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, stage if stage else "", (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

            cv2.imshow('Arnold Press Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()