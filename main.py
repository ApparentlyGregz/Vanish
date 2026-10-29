
import cv2
import mediapipe as mp
import time

from src.gesture_detector import GestureDetector
from src.invisibility import InvisibilityEffect

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
RunningMode = vision.RunningMode

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

gesture_detector = GestureDetector()
invisibility = InvisibilityEffect()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

start_time = time.monotonic()
last_timestamp = -1

try:
    with (
        vision.HandLandmarker.create_from_options(
            hand_options
        ) as hand_detector,
        vision.FaceLandmarker.create_from_options(
            face_options
        ) as face_detector
    ):
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Miroir AVANT détection pour garder
            # la gauche et la droite cohérentes
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp = max(
                timestamp, last_timestamp + 1
            )
            last_timestamp = timestamp

            hands = hand_detector.detect_for_video(
                image, timestamp
            )

            face = face_detector.detect_for_video(
                image, timestamp
            )

            visible, gesture = gesture_detector.detect(
                hands, face
            )
            # Image destinée à l'affichage
            output_frame = invisibility.apply(frame, visible)

            height, width = frame.shape[:2]

            # Afficher les doigts
            for hand in hands.hand_landmarks:
                for index in [8, 12]:
                    point = hand[index]
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame, (x, y),
                        7, (0, 0, 255), -1
                    )

            # Afficher les tempes
            if face.face_landmarks:
                landmarks = face.face_landmarks[0]

                for index in [127, 356]:
                    point = landmarks[index]

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame, (x, y),
                        9, (255, 0, 0), 2
                    )

            status = (
                "VISIBLE" if visible else "INVISIBLE"
            )

            color = (
                (0, 255, 0) if visible
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                f"VANISH: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            if gesture:
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # Afficher le flux modifié
            cv2.imshow("Vanish", output_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):
                invisibility.capture_background(frame)

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
