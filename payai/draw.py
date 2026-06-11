from PIL import ImageFont, ImageDraw, Image
from payai.config import CORES, LARGURA, ALTURA
from payai.logger import logger
import cv2
import numpy as np


def carregar_fontes():
    base = "C:/Windows/Fonts/"
    try:
        return {
            'logo_pay':  ImageFont.truetype(base + "segoeuib.ttf", 20),
            'logo_ai':   ImageFont.truetype(base + "segoeui.ttf",  20),
            'subtitulo': ImageFont.truetype(base + "segoeui.ttf",  11),
            'secao':     ImageFont.truetype(base + "segoeuib.ttf", 10),
            'corpo':     ImageFont.truetype(base + "segoeui.ttf",  12),
            'corpo_b':   ImageFont.truetype(base + "segoeuib.ttf", 12),
            'pequena':   ImageFont.truetype(base + "segoeui.ttf",  10),
            'badge':     ImageFont.truetype(base + "segoeuib.ttf", 11),
            'modo':      ImageFont.truetype(base + "segoeuib.ttf", 12),
            'contorno':  ImageFont.truetype(base + "segoeuib.ttf", 11),
            'hotkey':    ImageFont.truetype(base + "segoeuib.ttf", 13),
            'hotlabel':  ImageFont.truetype(base + "segoeui.ttf",  10),
        }
    except Exception as e:
        logger.warning(f"Segoe UI nao encontrada: {e}")
        f = ImageFont.load_default()
        return {k: f for k in ['logo_pay','logo_ai','subtitulo','secao','corpo',
                                'corpo_b','pequena','badge','modo','contorno',
                                'hotkey','hotlabel']}


FONTES = carregar_fontes()


# Correções: as funções abaixo convertem corretamente entre BGR (OpenCV) e RGB (PIL)
def cv2_para_pil(frame):
    # converte BGR -> RGB e retorna PIL Image
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_para_cv2(img):
    # recebe PIL Image, converte para array RGB e retorna BGR numpy array
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Funcoes auxiliares usadas por PayAI

def rect_r(draw, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
    if fill:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill)
    if outline:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=r, outline=outline, width=width)


def texto_c(draw, texto, cx, y, fonte, cor):
    w = fonte.getlength(texto)
    draw.text((cx - w / 2, y), texto, font=fonte, fill=cor)
