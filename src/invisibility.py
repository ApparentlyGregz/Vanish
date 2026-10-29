
import cv2
from pathlib import Path


class InvisibilityEffect:
    def __init__(self):
        # Chemin du projet Vanish
        project_dir = Path(__file__).resolve().parent.parent

        self.background_path = (
            project_dir / "assets" / "background.jpg"
        )

        self.background = None

        # Charger le fond s'il existe déjà
        if self.background_path.exists():
            self.background = cv2.imread(
                str(self.background_path)
            )

    def capture_background(self, frame):
        """Enregistre la chambre vide."""

        self.background_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.background = frame.copy()

        success = cv2.imwrite(
            str(self.background_path),
            self.background
        )

        if success:
            print("Arriere-plan enregistre !")
        else:
            print("Erreur lors de l'enregistrement")

    def apply(self, frame, visible):
        """Retourne l'image à afficher."""

        if visible or self.background is None:
            return frame.copy()

        height, width = frame.shape[:2]

        background = cv2.resize(
            self.background,
            (width, height)
        )

        return background.copy()
