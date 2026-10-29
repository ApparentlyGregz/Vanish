
import math
import time


class GestureDetector:
    def __init__(self):
        self.visible = True
        self.gesture_start = None
        self.current_gesture = None

        # Durée nécessaire pour valider un geste
        self.hold_duration = 0.5

        # Seuils à calibrer selon la webcam
        self.temple_threshold = 0.25
        self.fingers_threshold = 0.20

    def distance(self, p1, p2):
        return math.hypot(
            p1.x - p2.x,
            p1.y - p2.y
        )

    def detect(self, hand_results, face_results):
        gesture = None

        if face_results.face_landmarks:
            face = face_results.face_landmarks[0]

            # Points proches des côtés du visage
            temples = sorted(
                [face[127], face[356]],
                key=lambda p: p.x
            )

            # Image analysée en miroir :
            # gauche de l'image = gauche de l'utilisateur
            left_temple = temples[0]
            right_temple = temples[1]

            face_width = self.distance(
                left_temple, right_temple
            )

            if face_width > 0:
                for hand in hand_results.hand_landmarks:
                    index_tip = hand[8]
                    middle_tip = hand[12]

                    index_pip = hand[6]
                    middle_pip = hand[10]
                    wrist = hand[0]

                    # Vérifier que les doigts sont tendus
                    index_extended = (
                        self.distance(index_tip, wrist)
                        > self.distance(index_pip, wrist)
                    )

                    middle_extended = (
                        self.distance(middle_tip, wrist)
                        > self.distance(middle_pip, wrist)
                    )

                    fingers_close = (
                        self.distance(index_tip, middle_tip)
                        < face_width * self.fingers_threshold
                    )

                    if not (
                        index_extended
                        and middle_extended
                        and fingers_close
                    ):
                        continue

                    # Les deux doigts doivent être près
                    # de la même tempe
                    left_detected = all(
                        self.distance(p, left_temple)
                        < face_width * self.temple_threshold
                        for p in [index_tip, middle_tip]
                    )

                    right_detected = all(
                        self.distance(p, right_temple)
                        < face_width * self.temple_threshold
                        for p in [index_tip, middle_tip]
                    )

                    if left_detected:
                        gesture = "LEFT"
                        break

                    if right_detected:
                        gesture = "RIGHT"
                        break

        # Validation temporelle
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            self.gesture_start = (
                time.monotonic() if gesture else None
            )

        if gesture and self.gesture_start is not None:
            elapsed = time.monotonic() - self.gesture_start

            if elapsed >= self.hold_duration:
                if gesture == "LEFT":
                    self.visible = False
                elif gesture == "RIGHT":
                    self.visible = True

        return self.visible, gesture
