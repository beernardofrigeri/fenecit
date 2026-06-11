import threading
import time
import cv2
import easyocr
from queue import Empty
from payai.logger import logger
from payai.config import OCR_CONFIANCA_MINIMA, RESIZE_OCR, OCR_INTERVAL
from payai.utils import filtrar_valor_monetario, evitar_repeticao, pode_detectar, atualizar_ou_criar_contorno

reader = None
ocr_pronto = False
ocr_rodando = False
fila_ocr = None


def carregar_ocr():
    global reader, ocr_pronto
    try:
        logger.info("Carregando OCR...")
        reader = easyocr.Reader(['pt','es'], gpu=True, download_enabled=True)
        ocr_pronto = True
        logger.info("OCR carregado.")
    except Exception as e:
        logger.error(f"Erro carregando OCR: {e}")


def processar_valores(frame, estat, reader_local=None):
    global ocr_rodando
    global ocr_pronto
    if reader_local is None:
        reader_local = reader
    if reader_local is None:
        return

    ocr_rodando = True
    inicio_ocr = time.time()

    try:
        # redimensiona e converte
        small = cv2.resize(frame, RESIZE_OCR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        res = reader_local.readtext(gray, detail=1, paragraph=False)

        for (bbox, texto, conf) in res:
            if conf >= OCR_CONFIANCA_MINIMA:
                val = filtrar_valor_monetario(texto)
                if val:
                    # calcula contorno aproximado (escala)
                    sx = frame.shape[1] / RESIZE_OCR[0]
                    sy = frame.shape[0] / RESIZE_OCR[1]

                    tl = (int(bbox[0][0] * sx), int(bbox[0][1] * sy))
                    br = (int(bbox[2][0] * sx), int(bbox[2][1] * sy))

                    atualizar_ou_criar_contorno(tl, br, val, 'VALOR')

                    if evitar_repeticao(val) and pode_detectar(val):
                        logger.info(f"Valor: {val} (conf {conf:.2f})")
                        estat.registrar_deteccao('VALOR')
                        # devolve valor detectado para o chamador (que fará a fala)
                        return val

    except Exception as e:
        logger.error(f"Erro OCR: {e}")
        estat.registrar_erro()

    finally:
        ocr_rodando = False

    return None


class OCRThread(threading.Thread):
    def __init__(self, fila, estat):
        super().__init__(daemon=True)
        self.fila = fila
        self.estat = estat
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                frame = self.fila.get(timeout=0.1)
                val = processar_valores(frame, self.estat)
                if val:
                    # se houver valor, sinalizar ao chamador via log; fala deve ser feita fora
                    pass
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Erro loop OCR: {e}")

    def stop(self):
        self._stop = True
