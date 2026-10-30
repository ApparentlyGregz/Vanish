
import cv2
import pyvirtualcam


class VirtualCamera:
    def __init__(self, width, height, fps=30):
        self.camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
            backend="unitycapture"
        )

        print(
            f"Camera virtuelle active : {self.camera.device}"
        )

    def send(self, frame):
        # OpenCV : BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        self.camera.send(rgb_frame)
        self.camera.sleep_until_next_frame()

    def close(self):
        self.camera.close()
