
import cv2
import mediapipe as mp
import time
from pathlib import Path

from src.gesture_detector import GestureDetector
from src.invisibility import InvisibilityEffect
from src.virtual_camera import VirtualCamera


# ============================================
# CONFIGURATION
# ============================================

PROJECT_DIR = Path(__file__).resolve().parent

HAND_MODEL = PROJECT_DIR / "models" / "hand_landmarker.task"
FACE_MODEL = PROJECT_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30


# ============================================
# MEDIAPIPE
# ============================================

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
RunningMode = vision.RunningMode

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# ============================================
# INITIALISATION
# ============================================

gesture_detector = GestureDetector()
invisibility = InvisibilityEffect()

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

virtual_cam = None

start_time = time.monotonic()
last_timestamp = -1

print("Vanish started")
print("B : Capture background")
print("Q : Quit")
print("Left temple : Disappear")
print("Right temple : Reappear")


# ============================================
# MAIN LOOP
# ============================================

try:
    virtual_cam = VirtualCamera(
        width=width,
        height=height,
        fps=FPS
    )

    with (
        vision.HandLandmarker.create_from_options(
            hand_options
        ) as hand_detector,
        vision.FaceLandmarker.create_from_options(
            face_options
        ) as face_detector
    ):

        while True:

            # 1. Capture de la webcam
            success, frame = cap.read()

            if not success:
                print("Erreur de lecture webcam")
                break

            # Effet miroir pour la détection
            frame = cv2.flip(frame, 1)

            # 2. Conversion vers MediaPipe
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Timestamp croissant
            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp = max(
                timestamp,
                last_timestamp + 1
            )

            last_timestamp = timestamp

            # 3. Détection des mains et du visage
            hands = hand_detector.detect_for_video(
                image,
                timestamp
            )

            face = face_detector.detect_for_video(
                image,
                timestamp
            )

            # 4. Détection des gestes
            visible, gesture = gesture_detector.detect(
                hands,
                face
            )

            # 5. Application de l'invisibilité
            # Aucune annotation n'est ajoutée
            output_frame = invisibility.apply(
                frame,
                visible
            )

            # 6. Envoi vers Unity Capture / Teams
            virtual_cam.send(output_frame)

            # 7. Aperçu local propre
            cv2.imshow("Vanish", output_frame)

            # 8. Contrôles clavier
            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):
                invisibility.capture_background(frame)

            elif key == ord("q"):
                break


# ============================================
# CLEANUP
# ============================================

finally:
    cap.release()

    if virtual_cam is not None:
        virtual_cam.close()

    cv2.destroyAllWindows()

    print("Vanish stopped")

    #.\.venv\Scripts\python.exe main.py