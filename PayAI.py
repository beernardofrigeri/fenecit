import cv2
import easyocr
import asyncio
import edge_tts
import os
import time
import re
import logging
import threading
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from payai.camera import Camera
import pygame
import tempfile
from queue import Queue, Empty, Full
from payai.logger import logger
from payai.config import *
from payai.draw import carregar_fontes, FONTES, rect_r, texto_c, pil_para_cv2, cv2_para_pil
from payai.ocr import carregar_ocr, OCRThread, reader, ocr_pronto
from payai.speech import falar_texto

global ultimo_fps_tempo
global fps_atual

# ── LOGGING ───────────────────────────────────────────────────
# logging agora gerenciado por payai.logger
# logger importado do pacote payai

# ── OCR / VOZ ─────────────────────────────────────────────────
reader = None
ocr_pronto = False

# ── CAMERA ────────────────────────────────────────────────────
cap = None
camera_pronta = False

progresso_loading = 0.0

detector = cv2.QRCodeDetector()

# ── VARIAVEIS DE CONTROLE (carregadas de config.py) ───────────
LARGURA = LARGURA
ALTURA  = ALTURA

OCR_INTERVAL         = OCR_INTERVAL
SKIP_FRAMES          = SKIP_FRAMES
RESIZE_OCR           = RESIZE_OCR
OCR_CONFIANCA_MINIMA = OCR_CONFIANCA_MINIMA
CONTORNO_TEMPO_VIDA  = CONTORNO_TEMPO_VIDA

ultimo_tempo         = time.time()
texto_anterior       = ""
frame_count          = 0
ultimo_processamento = 0
contornos_ativos     = []
ultimos_detectados   = {}

fps_atual = 0
ultimo_fps_tempo = time.time()
tempo_ocr = 0
regioes_detectadas = 0

fila_ocr = Queue(maxsize=OCR_QUEUE_SIZE)

fala_lock            = threading.Lock()
ultima_fala          = ""

# Estado OCR
ocr_rodando = False

modo_atual = MODOS['AUTO']
idioma_atual = IDIOMAS['PT_BR']
# CORES, MODOS e IDIOMAS já são importados de payai.config

# ── FONTES ────────────────────────────────────────────────────
# def carregar_fontes():
#     base = "C:/Windows/Fonts/"
#     try:
#         return {
#             'logo_pay':  ImageFont.truetype(base + "segoeuib.ttf", 20),
#             'logo_ai':   ImageFont.truetype(base + "segoeui.ttf",  20),
#             'subtitulo': ImageFont.truetype(base + "segoeui.ttf",  11),
#             'secao':     ImageFont.truetype(base + "segoeuib.ttf", 10),
#             'corpo':     ImageFont.truetype(base + "segoeui.ttf",  12),
#             'corpo_b':   ImageFont.truetype(base + "segoeuib.ttf", 12),
#             'pequena':   ImageFont.truetype(base + "segoeui.ttf",  10),
#             'badge':     ImageFont.truetype(base + "segoeuib.ttf", 11),
#             'modo':      ImageFont.truetype(base + "segoeuib.ttf", 12),
#             'contorno':  ImageFont.truetype(base + "segoeuib.ttf", 11),
#             'hotkey':    ImageFont.truetype(base + "segoeuib.ttf", 13),
#             'hotlabel':  ImageFont.truetype(base + "segoeui.ttf",  10),
#         }
#     except Exception as e:
#         logger.warning(f"Segoe UI nao encontrada: {e}")
#         f = ImageFont.load_default()
#         return {k: f for k in ['logo_pay','logo_ai','subtitulo','secao','corpo',
#                                 'corpo_b','pequena','badge','modo','contorno',
#                                 'hotkey','hotlabel']}

# FONTES = carregar_fontes()

# ── UTILITARIOS DE DESENHO ────────────────────────────────────
def cv2_para_pil(frame):
    # compatibilidade com o módulo draw: converte BGR->RGB
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_para_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── CARREGAMENTO OCR ──────────────────────────────────────────
def carregar_ocr():

    global reader
    global ocr_pronto

    try:

        logger.info("Carregando OCR...")

        reader = easyocr.Reader(
            ['pt', 'es'],
            gpu=True,
            download_enabled=True
        )

        ocr_pronto = True

        logger.info("OCR carregado.")

    except Exception as e:

        logger.error(f"Erro carregando OCR: {e}")

        print(e)

# ── CARREGAMENTO CAMERA ───────────────────────────────────────
# A câmera será iniciada em thread para não bloquear a inicialização
cap = None
camera_pronta = False

_camera = Camera()
def _iniciar_camera():
    global cap, camera_pronta
    logger.info("Inicializando camera...")
    ok = _camera.start()
    if ok:
        cap = _camera.cap
        camera_pronta = True
        logger.info("Camera pronta.")
    else:
        logger.error("Erro ao acessar camera.")

threading.Thread(target=_iniciar_camera, daemon=True).start()

def rect_r(draw, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
    if fill:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill)
    if outline:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=r, outline=outline, width=width)

def texto_c(draw, texto, cx, y, fonte, cor):
    w = fonte.getlength(texto)
    draw.text((cx - w / 2, y), texto, font=fonte, fill=cor)


# ── CONTORNOS ─────────────────────────────────────────────────
def desenhar_contornos(draw, agora):
    for (top_left, bottom_right), texto, timestamp, tipo in contornos_ativos:
        restante = CONTORNO_TEMPO_VIDA - (agora - timestamp)
        alpha    = max(60, int(220 * restante / CONTORNO_TEMPO_VIDA))
        cor      = CORES['ACENTO'] if tipo == 'QRCODE' else CORES['AMARELO']

        draw.rectangle([top_left, bottom_right],
                       outline=(*cor, alpha), width=2)

        tx = top_left[0]
        ty = max(top_left[1] - 22, 44)
        bw = int(FONTES['contorno'].getlength(texto)) + 12
        rect_r(draw, tx, ty, tx + bw, ty + 18, r=3,
               fill=(*cor, min(alpha, 210)))
        draw.text((tx + 6, ty + 3), texto,
                  font=FONTES['contorno'], fill=CORES['PRETO'])


# ── INTERFACE ─────────────────────────────────────────────────
def desenhar_interface(frame, estatisticas, agora):

    img  = cv2_para_pil(frame)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ── HEADER ────────────────────────────────────────────────
    draw.rectangle([0, 0, W, 42], fill=CORES['BG'])
    draw.line(
        [(0, 42), (W, 42)], 
        fill=CORES['BORDA'],
        width=1
    ) 

    # Logo
    lp = int(FONTES['logo_pay'].getlength("PAY"))
    draw.text((14, 11), "PAY", font=FONTES['logo_pay'], fill=CORES['ACENTO'])
    draw.text((14 + lp + 4, 11), "AI", font=FONTES['logo_ai'], fill=CORES['TEXTO_PRIM'])

    # Separador vertical
    draw.line(
        [(14 + lp + 4 + 26, 13), (14 + lp + 4 + 26, 30)],
        fill=CORES['BORDA'],
        width=1
    )
    

    # Subtitulo
    draw.text(
        (14 + lp + 4 + 33, 14),
        "Sistema Inteligente",
        font=FONTES['subtitulo'],
        fill=CORES['TEXTO_SEC']
    )

     # ── BADGES SUPERIORES ──────────────────────────────────
    modos_cfg = {
        0: ("AUTO",    CORES['ACENTO']),
        1: ("VALORES", CORES['AMARELO']),
        2: ("QR CODE", CORES['AZUL']),
    }

    modo_label, cor_modo = modos_cfg.get(
        modo_atual,
        ("AUTO", CORES['ACENTO'])
    )

    idioma_label = (
        'PT-BR'
        if idioma_atual == IDIOMAS['PT_BR']
        else 'ES-CO'
    )

    # ── BADGE MODO ─────────────────────────
    modo_x1 = W - 235
    modo_y1 = 8
    modo_x2 = W - 145
    modo_y2 = 32

    rect_r(
        draw,
        modo_x1,
        modo_y1,
        modo_x2,
        modo_y2,
        r=6,
        fill=CORES['CARD_BG'],
        outline=cor_modo,
        width=1
    )

    texto_c(
        draw,
        modo_label,
        (modo_x1 + modo_x2) // 2,
        modo_y1 + 5,
        FONTES['modo'],
        cor_modo
    )

    # ── BADGE IDIOMA ───────────────────────
    idioma_x1 = W - 135
    idioma_y1 = 8
    idioma_x2 = W - 20
    idioma_y2 = 32

    rect_r(
        draw,
        idioma_x1,
        idioma_y1,
        idioma_x2,
        idioma_y2,
        r=6,
        fill=CORES['CARD_BG'],
        outline=CORES['AZUL'],
        width=1
    )

    texto_c(
        draw,
        idioma_label,
        (idioma_x1 + idioma_x2) // 2,
        idioma_y1 + 5,
        FONTES['badge'],
        CORES['TEXTO_PRIM']
    )

    # ── STATS ────────────────────────────────────────────────
    if estatisticas:

        stats = estatisticas.obter_estatisticas()

        itens = [
            ("Valores",  str(stats['valores_detectados'])),
            ("QR Codes", str(stats['qrcodes_detectados'])),
            ("Taxa",     f"{stats['detectoes_por_minuto']:.1f}/min"),

            ("FPS",      f"{fps_atual:.1f}"),
            ("OCR",      f"{tempo_ocr:.0f}ms"),
            ("Regioes",  str(regioes_detectadas)),
        ]       

        sy = 50

        for label, valor in itens:

            draw.text(
                (14, sy),
                f"{label}:",
                font=FONTES['pequena'],
                fill=CORES['TEXTO_SEC']
            )

            lw = int(FONTES['pequena'].getlength(f"{label}:")) + 6

            draw.text(
                (14 + lw, sy),
                valor,
                font=FONTES['corpo_b'],
                fill=CORES['TEXTO_PRIM']
            )

            sy += 16

    # ── CONTORNOS ────────────────────────────────────────────
    desenhar_contornos(draw, agora)

    # ── FOOTER ───────────────────────────────────────────────
    footer_y = H - 72

    draw.rectangle(
        [0, footer_y, W, H],
        fill=CORES['BG']
    )

    draw.line(
        [(0, footer_y), (W, footer_y)],
        fill=CORES['BORDA'],
        width=1
    )

    botoes = [
        ("ESC", "SAIR"),
        ("V", "VALORES"),
        ("Q", "QR CODE"),
        ("A", "AUTO"),
        ("I", "IDIOMA"),
        ("R", "REPETIR"),
        ("S", "PRINT")
    ]

    card_w  = 72
    gap     = 4
    total_w = len(botoes) * card_w + (len(botoes) - 1) * gap
    bx_ini  = (W - total_w) // 2

    for i, (tecla, desc) in enumerate(botoes):

        cx1 = bx_ini + i * (card_w + gap)
        cx2 = cx1 + card_w
        cy1 = footer_y + 8
        cy2 = H - 8

        ativo = (
            (tecla == "A" and modo_atual == 0) or
            (tecla == "V" and modo_atual == 1) or
            (tecla == "Q" and modo_atual == 2)
        )

        cor_fill  = CORES['ACENTO_DIM'] if ativo else CORES['CARD_BG']
        cor_borda = CORES['ACENTO'] if ativo else CORES['BORDA']

        rect_r(
            draw,
            cx1,
            cy1,
            cx2,
            cy2,
            r=6,
            fill=cor_fill,
            outline=cor_borda,
            width=1
        )

        # Tecla
        tw = FONTES['hotkey'].getlength(tecla)

        draw.text(
            (cx1 + (card_w - tw) / 2, cy1 + 5),
            tecla,
            font=FONTES['hotkey'],
            fill=CORES['ACENTO'] if ativo else CORES['TEXTO_PRIM']
        )

        # Label
        dw = FONTES['hotlabel'].getlength(desc)

        draw.text(
            (cx1 + (card_w - dw) / 2, cy2 - 17),
            desc,
            font=FONTES['hotlabel'],
            fill=CORES['ACENTO'] if ativo else CORES['TEXTO_SEC']
        )

    return pil_para_cv2(img)

# ── CLASSES ───────────────────────────────────────────────────
class Estatisticas:
    def __init__(self):
        self.valores_detectados = 0
        self.qrcodes_detectados = 0
        self.erros_ocr          = 0
        self.inicio             = time.time()

    def registrar_deteccao(self, tipo):
        if tipo == 'VALOR':    
            self.valores_detectados += 1
        elif tipo == 'QRCODE': 
            self.qrcodes_detectados += 1

    def registrar_erro(self):
        self.erros_ocr += 1

    def obter_estatisticas(self):
        t     = time.time() - self.inicio
        total = self.valores_detectados + self.qrcodes_detectados
        return {
            'tempo_execucao':       t,
            'valores_detectados':   self.valores_detectados,
            'qrcodes_detectados':   self.qrcodes_detectados,
            'erros_ocr':            self.erros_ocr,
            'detectoes_por_minuto': total / (t / 60) if t > 0 else 0,
        }

# ── CONVERSAO NUMERICA ────────────────────────────────────────

def _num_pt(n):
    if n == 0: 
        return "zero"
    if n <= 9:
        return ['','um','dois','tres','quatro','cinco','seis','sete','oito','nove'][n]
    if n <= 19:
        return ['dez','onze','doze','treze','quatorze','quinze',
                'dezesseis','dezessete','dezoito','dezenove'][n-10]
    if n <= 99:
        dez = ['','','vinte','trinta','quarenta','cinquenta',
               'sessenta','setenta','oitenta','noventa']
        d, u = n // 10, n % 10
        return dez[d] if u == 0 else f"{dez[d]} e {_num_pt(u)}"
    if n <= 999:
        if n == 100: 
            return "cem"
        c = ['','cento','duzentos','trezentos','quatrocentos','quinhentos',
             'seiscentos','setecentos','oitocentos','novecentos']
        cent, r = n // 100, n % 100
        return c[cent] + (" e " + _num_pt(r) if r else "")
    return f"{n:,}".replace(",", ".")

# ── CONVERSAO ESPANHOL ───────────────────────────────────────
def numero_es(n):

    unidades = [
        'cero', 'uno', 'dos', 'tres', 'cuatro',
        'cinco', 'seis', 'siete', 'ocho', 'nueve'
    ]

    especiais = {
        10: 'diez',
        11: 'once',
        12: 'doce',
        13: 'trece',
        14: 'catorce',
        15: 'quince'
    }

    dezenas = {
        20: 'veinte',
        30: 'treinta',
        40: 'cuarenta',
        50: 'cincuenta',
        60: 'sesenta',
        70: 'setenta',
        80: 'ochenta',
        90: 'noventa'
    }

    if n < 10:
        return unidades[n]

    if n in especiais:
        return especiais[n]

    if n < 20:
        return 'dieci' + unidades[n - 10]

    if n < 30:
        return 'veinti' + unidades[n - 20]

    if n < 100:
        d = (n // 10) * 10
        r = n % 10

        if r == 0:
            return dezenas[d]

        return f"{dezenas[d]} y {unidades[r]}"

    return str(n)

# ── VOZ ───────────────────────────────────────────────────────
async def falar_edge(texto, voz="es-CO-GonzaloNeural"):

    if not pygame.mixer.get_init():
        pygame.mixer.init()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        caminho = fp.name

    communicate = edge_tts.Communicate(texto, voz)

    await communicate.save(caminho)

    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

    pygame.mixer.music.unload()

    os.remove(caminho)

def falar_texto(texto):

    global ultima_fala
    ultima_fala = texto

    def _falar():

        with fala_lock:

            try:

                tf = texto

                mr = re.search(r'(\d+)\s*reais', texto.lower())
                mc = re.search(r'(\d+)\s*centavos', texto.lower())

                # ── PORTUGUES ─────────────────────
                if idioma_atual == IDIOMAS['PT_BR']:

                    if mr:
                        n = int(mr.group(1))
                        tf = f"{_num_pt(n)} reais"

                        if mc:
                            tf += f" e {_num_pt(int(mc.group(1)))} centavos"

                    elif mc:
                        tf = f"{_num_pt(int(mc.group(1)))} centavos"

                # ── ESPANHOL ──────────────────────
                elif idioma_atual == IDIOMAS['ES_CO']:

                    if mr:
                        n = int(mr.group(1))
                        tf = f"{numero_es(n)} pesos colombianos"

                    elif mc:
                        tf = f"{numero_es(int(mc.group(1)))} centavos"

                    tf = tf.replace(
                        'QR Code detectado',
                        'Código QR detectado'
                    )

                    tf = tf.replace(
                        'Modo automatico ativado',
                        'Modo automático activado'
                    )

                    tf = tf.replace(
                        'Modo valores ativado',
                        'Modo valores activado'
                    )

                    tf = tf.replace(
                        'Modo QR Code ativado',
                        'Modo código QR activado'
                    )

                    tf = tf.replace(
                        'Screenshot salvo',
                        'Captura guardada'
                    )

                if idioma_atual == IDIOMAS['ES_CO']:
                    asyncio.run(
                        falar_edge(
                            tf,
                            "es-CO-GonzaloNeural"
                        )
                    )

                else:

                    asyncio.run(
                        falar_edge(
                            tf,
                            "pt-BR-AntonioNeural"
                        )
                    )

            except Exception as e:
                logger.error(f"Erro na fala: {e}")

    threading.Thread(target=_falar, daemon=True).start()
# ── FILTRO MONETARIO ──────────────────────────────────────────
def validar_valor(val):
    try:
        p = val.split(',')
        if len(p) != 2: 
            return False
        i, d = p[0].replace('.', ''), p[1]
        if len(d) != 2 or not d.isdigit() or not i.isdigit(): 
            return False
        return 0 <= float(f"{i}.{d}") <= 10000
    except Exception:
        return False

def formatar_fala(val):
    p    = val.split(',')
    i, c = int(p[0].replace('.', '')), int(p[1])
    if c == 0: 
        return f"{i} reais"
    if i == 0: 
        return f"{c} centavos"
    return f"{i} reais e {c} centavos"

def filtrar_valor_monetario(texto):
    padroes = [
        r'R\s*[\$\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*R',
        r'valor\s*[\:\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'total\s*[\:\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d+,\d{2})\s*(?:reais|R\$)',
        r'(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d+,\d{2})',
    ]
    for p in padroes:
        m = re.search(p, texto, re.IGNORECASE)
        if m and validar_valor(m.group(1)):
            return formatar_fala(m.group(1))
    return None

def evitar_repeticao(texto, minimo=3):
    global ultimo_tempo, texto_anterior
    agora = time.time()
    if texto == texto_anterior and (agora - ultimo_tempo) < minimo:
        return False
    texto_anterior, ultimo_tempo = texto, agora
    return True

def pode_detectar(texto, cooldown=5):

    agora = time.time()

    if texto in ultimos_detectados:

        if agora - ultimos_detectados[texto] < cooldown:
            return False

    ultimos_detectados[texto] = agora
    return True

def atualizar_contornos():
    global contornos_ativos
    agora = time.time()
    contornos_ativos = [
        item for item in contornos_ativos
        if (agora - item[2]) < CONTORNO_TEMPO_VIDA
    ]

def atualizar_ou_criar_contorno(tl, br, texto, tipo):

    global contornos_ativos

    agora = time.time()

    for i, item in enumerate(contornos_ativos):

        (_, _), txt, _, tp = item

        if txt == texto and tp == tipo:

            contornos_ativos[i] = (
                (tl, br),
                texto,
                agora,
                tipo
            )

            return

    contornos_ativos.append(
        ((tl, br), texto, agora, tipo)
    )


# ── PROCESSAMENTO ─────────────────────────────────────────────
def detectar_regioes_texto(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # melhora contraste
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # detecta bordas
    edges = cv2.Canny(blur, 80, 150)

    # junta áreas próximas
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3)
    )

    dilatada = cv2.dilate(
        edges,
        kernel,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        dilatada,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regioes = []

    for cnt in contornos:

        x, y, w, h = cv2.boundingRect(cnt)

        # filtros contra lixo
        if w < 60 or h < 20:
            continue

        if w > 500 or h > 200:
            continue

        area = w * h

        if area < 1500:
            continue

        roi = frame[y:y+h, x:x+w]

        regioes.append((roi, (x, y, w, h)))

    return regioes

def processar_valores(frame, estat):
    
    if reader is None:
        return

    global ocr_rodando
    global tempo_ocr
    global regioes_detectadas
    ocr_rodando = True
    inicio_ocr = time.time()

    try:
        regioes = detectar_regioes_texto(frame)

        regioes_detectadas = len(regioes)
        for roi, (rx, ry, rw, rh) in regioes:
            small = cv2.resize(
                roi,
                RESIZE_OCR
            )

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY
            )

            res = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                batch_size=1,
                width_ths=2.0,
                height_ths=2.0
            )

            for (bbox, texto, conf) in res:
            
                if conf >= OCR_CONFIANCA_MINIMA:
                
                    val = filtrar_valor_monetario(texto)

                    if val:

                        sx = LARGURA / RESIZE_OCR[0]
                        sy = ALTURA  / RESIZE_OCR[1]
    
                        tl = (
                            int(bbox[0][0] * sx) + rx,
                            int(bbox[0][1] * sy) + ry
                        )
                        br = (
                            int(bbox[2][0] * sx) + rx,
                            int(bbox[2][1] * sy) + ry
                        )
    
                        valor_visual = texto.replace("R$", "").strip()
    
                        if idioma_atual == IDIOMAS['ES_CO']:
                            texto_contorno = f"COL$ {valor_visual} pesos colombianos"
                        else:
                            texto_contorno = f"R$ {valor_visual} reais"
    
                        # ── SEMPRE atualiza o contorno ──
                        atualizar_ou_criar_contorno(
                            tl,
                            br,
                            texto_contorno,
                            'VALOR'
                        )
    
                        # ── fala só uma vez ──
                        if evitar_repeticao(val) and pode_detectar(val):
                        
                            logger.info(f"Valor: {val} (conf {conf:.2f})")
    
                            estat.registrar_deteccao('VALOR')
    
                            falar_texto(val)

    except Exception as e:
        logger.error(f"Erro OCR: {e}")
        estat.registrar_erro()

    finally:
        tempo_ocr = (
            time.time() - inicio_ocr
        ) * 1000

    ocr_rodando = False

def loop_ocr(estat):

    while True:

        try:

            frame = fila_ocr.get(timeout=0.1)

            processar_valores(frame, estat)

        except Empty:

            pass

        except Exception as e:

            logger.error(f"Erro loop OCR: {e}")
        

def processar_qrcode(frame, estat):

    try:

        data, bbox, _ = detector.detectAndDecode(frame)

    except cv2.error as e:

        logger.warning(f"Erro QRCodeDetector: {e}")

        return

    if data and data.strip():
        # fallback behavior: log data and notify
        logger.info(f"QR: {data}")
        estat.registrar_deteccao('QRCODE')
        falar_texto("QR Code detectado", idioma_atual, None)

        if bbox is not None:

            pts = bbox.astype(int).reshape(-1, 2)

            tl = (
                int(pts[:, 0].min()),
                int(pts[:, 1].min())
            )

            br = (
                int(pts[:, 0].max()),
                int(pts[:, 1].max())
            )

            txt = (
                data[:18] + "..."
                if len(data) > 18
                else data
            )

            contornos_ativos.append(
                (
                    (tl, br),
                    f"QR: {txt}",
                    time.time(),
                    'QRCODE'
                )
            )


# ── INICIALIZACAO ─────────────────────────────────────────────
estatisticas = Estatisticas()

print(f"[INFO] Aponte a camera para o visor da maquininha ou QR Code...")
logger.info(f"Sistema PayAI iniciado - {LARGURA}x{ALTURA}")

# ── THREAD DE CARREGAMENTO OCR ───────────────────────────────
threading.Thread(
    target=carregar_ocr,
    daemon=True
).start()

# ── THREAD CAMERA ────────────────────────────────────────────
# threading.Thread(
#     target=carregar_camera,
#     daemon=True
# ).start()

def _iniciar_camera_thread():
    global cap, camera_pronta
    ok = _camera.start()
    if ok:
        cap = _camera.cap
        camera_pronta = True

threading.Thread(
    target=_iniciar_camera_thread,
    daemon=True
).start()

# ── LOOP PRINCIPAL ────────────────────────────────────────────
_ocr_thread = OCRThread(fila_ocr, estatisticas)
_ocr_thread.start()

try:
    while True:
                
    # ── SPLASH SCREEN ───────────────────────────────────
        if not camera_pronta or not ocr_pronto:

            splash = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)

            splash[:] = CORES['BG']

            img = cv2_para_pil(splash)

            draw = ImageDraw.Draw(img)

            texto_c(
                draw,
                "PAYAI",
                LARGURA // 2,
                165,
                FONTES['logo_pay'],
                CORES['ACENTO']
            )

            texto_c(
                draw,
                "Inicializando sistema...",
                LARGURA // 2,
                225,
                FONTES['corpo_b'],
                CORES['TEXTO_PRIM']
            )

            status = []

            if not camera_pronta:
                status.append("Camera")

            if not ocr_pronto:
                status.append("OCR")

            mensagens_loading = [

                "Inicializando componentes",
                "Preparando sistema",
                "Carregando interface",
                "Verificando modulos",
                "Inicializando camera",
                "Preparando reconhecimento",
                "Otimizando OCR",
                "Configurando acessibilidade",
                "Preparando sistema de voz",
                "Carregando deteccao inteligente",
                "Verificando desempenho",
                "Sincronizando componentes",
                "Iniciando visao computacional",
                "Configurando analise inteligente",
                "Inicializando assistente visual",
                "Preparando leitura automatica",
                "Sincronizando reconhecimento",
                "Aplicando melhorias de desempenho",
                "Verificando integridade do sistema",
                "Otimizando inicializacao",
                "Preparando captura de imagem",
                "Ativando componentes principais",
                "Preparando ambiente de execucao",

            ]

            indice_msg = int(
                time.time() * 0.45
            ) % len(mensagens_loading)

            texto_status = mensagens_loading[indice_msg]

            texto_status += f" | CAM:{camera_pronta} OCR:{ocr_pronto}"

            texto_c(
                draw,
                texto_status,
                LARGURA // 2,
                255,
                FONTES['pequena'],
                CORES['TEXTO_SEC']
            )

            # Barra simples
            draw.rounded_rectangle(
                [170, 305, 470, 317],
                radius=5,
                fill=CORES['CARD_BG']
            )

            # ── PROGRESSO SUAVE ───────────────────────

            alvo = 0.90

            if camera_pronta and ocr_pronto:
                alvo = 1.0

            if progresso_loading < alvo:
                progresso_loading += 0.025

            progresso_loading = min(
                progresso_loading,
                alvo
            )

            largura = int(300 * progresso_loading)

            draw.rounded_rectangle(
                [170, 305, 170 + largura, 317],
                radius=5,
                fill=CORES['ACENTO']
            )

             # ── BRILHO SUAVE E CONTINUO ───────────

            if largura > 80:

                barra_x1 = 170
                barra_x2 = 170 + largura

                brilho_total = largura

                brilho = (
                    time.time() * 70
                ) % brilho_total

                shine_x = barra_x1 + brilho

                brilho_largura = 28

                # Evita sair seco da barra
                if shine_x < barra_x2:

                    # Fade perto da entrada
                    entrada = min(
                        1.0,
                        max(
                            0,
                            (shine_x - barra_x1) / 25
                        )
                    )

                    # Fade perto da saída
                    saida = min(
                        1.0,
                        max(
                            0,
                            (barra_x2 - shine_x) / 25
                        )
                    )

                    intensidade = min(
                        entrada,
                        saida
                    )

                    branco = int(
                        140 + (115 * intensidade)
                    )

                    draw.rounded_rectangle(
                        [
                            shine_x - 2,
                            307,
                            shine_x + brilho_largura,
                            315
                        ],
                        radius=4,
                        fill=(branco, branco, branco)
                    )

            splash = pil_para_cv2(img)

            cv2.imshow(
                "PayAI - Sistema Inteligente",
                splash
            )

            cv2.waitKey(1)

            continue

        # ── CAMERA NORMAL ───────────────────────────────────
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0:
            agora_fps = time.time()

            fps_atual = 10 / (
                agora_fps - ultimo_fps_tempo
            )

            ultimo_fps_tempo = agora_fps
        
        agora = time.time()

        atualizar_contornos()

        if modo_atual in (MODOS['AUTO'], MODOS['VALORES']):
            if (
                agora - ultimo_processamento > OCR_INTERVAL and
                frame_count % SKIP_FRAMES == 0
            ):
                ultimo_processamento = agora
                try:
                    fila_ocr.put_nowait(frame.copy())
                except Full:
                    pass

        if modo_atual in (MODOS['AUTO'], MODOS['QRCODE']):
            processar_qrcode(frame, estatisticas)

        frame_final = desenhar_interface(frame, estatisticas, agora)

        cv2.imshow("PayAI - Sistema Inteligente", frame_final)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            logger.info("Encerrado pelo usuario")
            break
        elif key in (ord('v'), ord('V')):
            modo_atual = MODOS['VALORES']
            falar_texto("Modo valores ativado", idioma_atual, None)
        elif key in (ord('q'), ord('Q')):
            modo_atual = MODOS['QRCODE']
            falar_texto("Modo QR Code ativado", idioma_atual, None)
        elif key in (ord('a'), ord('A')):
            modo_atual = MODOS['AUTO']
            falar_texto("Modo automatico ativado", idioma_atual, None)
        elif key in (ord('i'), ord('I')):
            if idioma_atual == IDIOMAS['PT_BR']:
                idioma_atual = IDIOMAS['ES_CO']
                falar_texto('Idioma español activado', idioma_atual, None)
            else:
                idioma_atual = IDIOMAS['PT_BR']
                falar_texto('Idioma portugues ativado', idioma_atual, None)
        elif key in (ord('s'), ord('S')):
            nome = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(nome, frame_final)
            logger.info(f"Screenshot: {nome}")
            falar_texto("Screenshot salvo", idioma_atual, None)
        elif key in (ord('r'), ord('R')):
            if ultima_fala:
                falar_texto(ultima_fala, idioma_atual, None)

except Exception as e:
    logger.error(f"Erro critico: {e}")

finally:
    if cap:
        try:
            cap.release()
        except Exception:
            pass
    else:
        # se camera for gerenciada pela classe
        try:
            _camera.release()
        except Exception:
            pass

    cv2.destroyAllWindows()
    logger.info(f"Estatisticas finais: {estatisticas.obter_estatisticas()}")
    logger.info("Sistema PayAI finalizado")
    # parar thread OCR
    try:
        _ocr_thread.stop()
    except Exception:
        pass