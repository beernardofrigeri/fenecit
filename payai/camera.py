import cv2
from payai.config import LARGURA, ALTURA
from payai.logger import logger

class Camera:
    def __init__(self, device=0, width=LARGURA, height=ALTURA, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        logger.info("Inicializando camera...")
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            # nem todos drivers suportam buffersize
            pass

        if not self.cap.isOpened():
            logger.error("Erro ao acessar camera.")
            return False

        logger.info("Camera pronta.")
        return True

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
