# ── RESOLUCAO ─────────────────────────────
LARGURA = 640
ALTURA = 480

# ── OCR ───────────────────────────────────
OCR_INTERVAL = 0.7
SKIP_FRAMES = 8
RESIZE_OCR = (320, 240)
OCR_CONFIANCA_MINIMA = 0.7

# ── CONTORNOS ─────────────────────────────
CONTORNO_TEMPO_VIDA = 3.0

# ── FILA OCR ──────────────────────────────
OCR_QUEUE_SIZE = 1

# ── MODOS ─────────────────────────────────
MODOS = {
    'AUTO': 0,
    'VALORES': 1,
    'QRCODE': 2
}

# ── IDIOMAS ───────────────────────────────
IDIOMAS = {
    'PT_BR': 0,
    'ES_CO': 1
}

# ── CORES ─────────────────────────────────
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