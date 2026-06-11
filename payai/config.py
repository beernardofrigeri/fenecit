"""
Config centralizada para o pacote payai.
Importa valores de config.py do projeto quando possivel.
"""
import json
import os
from pathlib import Path

# Valores padrao (fallback)
LARGURA = 640
ALTURA = 480

OCR_INTERVAL = 0.7
SKIP_FRAMES = 8
RESIZE_OCR = (320, 240)
OCR_CONFIANCA_MINIMA = 0.7
CONTORNO_TEMPO_VIDA = 3.0
OCR_QUEUE_SIZE = 1

MODOS = {
    'AUTO': 0,
    'VALORES': 1,
    'QRCODE': 2
}

IDIOMAS = {
    'PT_BR': 0,
    'ES_CO': 1
}

CORES = {
    'BRANCO':       (255, 255, 255),
    'PRETO':        (0,   0,   0),
    'BG':           (0,  0,  0),
    'CARD_BG':      (30,  35,  44),
    'BORDA':        (48,  54,  64),
    'TEXTO_SEC':    (130, 140, 160),
    'TEXTO_PRIM':   (220, 225, 235),
    'ACENTO':       (0,   200, 180),
    'ACENTO_DIM':   (0,   60,  54),
    'AMARELO':      (230, 180,   0),
    'AZUL':         (80,  150, 255),
    'VERDE_STATUS': (50,  200, 100),
}

# Tenta carregar config.json se existir
_cfg_path = Path(os.getcwd()) / 'config.json'
if _cfg_path.exists():
    try:
        with open(_cfg_path, 'r', encoding='utf-8') as fp:
            j = json.load(fp)

            o = j.get('ocr', {})
            # Não sobrescrever LARGURA/ALTURA com o tamanho do OCR
            OCR_INTERVAL = o.get('intervalo', OCR_INTERVAL)
            SKIP_FRAMES = o.get('skip_frames', SKIP_FRAMES)
            RESIZE_OCR = tuple(o.get('tamanho_redimensionado', RESIZE_OCR))
            OCR_CONFIANCA_MINIMA = o.get('confianca_minima', OCR_CONFIANCA_MINIMA)

            # interface
            CONTORNO_TEMPO_VIDA = j.get('interface', {}).get('tempo_vida_contorno', CONTORNO_TEMPO_VIDA)
            # opcionalmente carregar largura/altura se o arquivo especificar
            LARGURA = j.get('interface', {}).get('largura', LARGURA)
            ALTURA = j.get('interface', {}).get('altura', ALTURA)

    except Exception:
        # falha ao carregar config.json, usar valores padrao
        pass
