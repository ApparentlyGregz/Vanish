
import cv2
import mediapipe as mp
import time

# Configuration de MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Initialisation de la webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

start_time = time.monotonic()
last_timestamp = -1

with HandLandmarker.create_from_options(options) as detector:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Conversion BGR vers RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Timestamp croissant obligatoire en mode VIDEO
        timestamp = int((time.monotonic() - start_time) * 1000)
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(
            mp_image, timestamp
        )

        height, width = frame.shape[:2]

        # Affichage des points des mains
        for hand in result.hand_landmarks:
            for i, landmark in enumerate(hand):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                color = (0, 255, 0)

                # Index (8) et majeur (12) en rouge
                if i in [8, 12]:
                    color = (0, 0, 255)

                cv2.circle(frame, (x, y), 5, color, -1)

        # Image en miroir pour l'affichage
        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "VANISH - HAND TRACKING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Vanish", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
