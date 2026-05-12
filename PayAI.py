import cv2
import easyocr
import pyttsx3
import time
import re
import logging
import threading
from collections import deque
import numpy as np
import gc
from functools import lru_cache
import json
import os
from PIL import ImageFont, ImageDraw, Image

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('payai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── OCR / VOZ ─────────────────────────────────────────────────
reader = easyocr.Reader(['pt', 'en', 'es'], gpu=False)

engine = pyttsx3.init()
engine.setProperty('rate', 130)

voices = engine.getProperty('voices')

voz_ptbr = None
voz_esco = None

for voice in voices:
    nome = voice.name.lower()

    # Português
    if any(t in nome for t in ['portugues', 'brazil', 'brasil', 'pt', 'br']):
        voz_ptbr = voice

    # Espanhol
    if any(t in nome for t in ['spanish', 'español', 'espanol', 'colombia', 'es']):
        voz_esco = voice

if voz_ptbr:
    logger.info(f"Voz PT-BR configurada: {voz_ptbr.name}")
else:
    logger.warning("Voz PT-BR nao encontrada")

if voz_esco:
    logger.info(f"Voz ES-CO configurada: {voz_esco.name}")
else:
    logger.warning("Voz ES-CO nao encontrada")

# ── CAMERA ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    logger.error("Erro ao acessar a camera.")
    exit()

detector = cv2.QRCodeDetector()

# ── VARIAVEIS DE CONTROLE ─────────────────────────────────────
ultimo_tempo         = time.time()
texto_anterior       = ""
historico_valores    = deque(maxlen=8)
frame_count          = 0
ultimo_processamento = 0
contornos_ativos     = []
CONTORNO_TEMPO_VIDA  = 3.0
OCR_INTERVAL         = 0.3
SKIP_FRAMES          = 5
RESIZE_OCR           = (320, 240)
fala_lock            = threading.Lock()
ultima_fala = ""

MODOS = {'AUTO': 0, 'VALORES': 1, 'QRCODE': 2}
modo_atual = MODOS['AUTO']

IDIOMAS = {
    'PT_BR': 0,
    'ES_CO': 1
}

idioma_atual = IDIOMAS['PT_BR']

# ── PALETA ────────────────────────────────────────────────────
CORES = {
    'BRANCO':       (255, 255, 255),
    'PRETO':        (0,   0,   0),
    'BG':           (13,  17,  23),
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

# ── IDIOMA ────────────────────────────────────────────────────
def configurar_idioma():
    global idioma_atual

    if idioma_atual == IDIOMAS['PT_BR']:
        if voz_ptbr:
            engine.setProperty('voice', voz_ptbr.id)

    elif idioma_atual == IDIOMAS['ES_CO']:
        if voz_esco:
            engine.setProperty('voice', voz_esco.id)

# ── FONTES ────────────────────────────────────────────────────
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

# ── UTILITARIOS DE DESENHO ────────────────────────────────────
def cv2_para_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_para_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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
def desenhar_interface(frame, estatisticas):

    img  = cv2_para_pil(frame)
    draw = ImageDraw.Draw(img, 'RGBA')
    W, H = img.size

    # ── HEADER ────────────────────────────────────────────────
    draw.rectangle([0, 0, W, 42], fill=(*CORES['BG'], 220))
    draw.line([(0, 42), (W, 42)], fill=(*CORES['BORDA'], 255), width=1)

    # Logo
    lp = int(FONTES['logo_pay'].getlength("PAY"))
    draw.text((14, 11), "PAY", font=FONTES['logo_pay'], fill=CORES['ACENTO'])
    draw.text((14 + lp + 4, 11), "AI", font=FONTES['logo_ai'], fill=CORES['TEXTO_PRIM'])

    # Separador vertical
    draw.line(
        [(14 + lp + 4 + 26, 13), (14 + lp + 4 + 26, 30)],
        fill=(*CORES['BORDA'], 255),
        width=1
    )

    # Subtitulo
    draw.text(
        (14 + lp + 4 + 33, 14),
        "Sistema Inteligente",
        font=FONTES['subtitulo'],
        fill=CORES['TEXTO_SEC']
    )

    # ── BADGE MODO ───────────────────────────────────────────
    modos_cfg = {
        0: ("AUTO",    CORES['ACENTO']),
        1: ("VALORES", CORES['AMARELO']),
        2: ("QR CODE", CORES['AZUL']),
    }

    modo_label, cor_modo = modos_cfg.get(
        modo_atual,
        ("AUTO", CORES['ACENTO'])
    )

    rect_r(
        draw,
        W - 250,
        9,
        W - 170,
        33,
        r=5,
        fill=(*cor_modo, 22),
        outline=cor_modo,
        width=1
    )

    texto_c(
        draw,
        modo_label,
        W - 210,
        13,
        FONTES['modo'],
        cor_modo
    )

    # ── BADGE IDIOMA ─────────────────────────────────────────
    idioma_label = (
        'PT-BR'
        if idioma_atual == IDIOMAS['PT_BR']
        else 'ES-CO'
    )

    rect_r(
        draw,
        W - 160,
        9,
        W - 12,
        33,
        r=5,
        fill=(*CORES['AZUL'], 25),
        outline=CORES['AZUL'],
        width=1
    )

    texto_c(
        draw,
        idioma_label,
        W - 86,
        13,
        FONTES['badge'],
        CORES['AZUL']
    )

    # ── STATS ────────────────────────────────────────────────
    if estatisticas:

        stats = estatisticas.obter_estatisticas()

        itens = [
            ("Valores",  str(stats['valores_detectados'])),
            ("QR Codes", str(stats['qrcodes_detectados'])),
            ("Taxa",     f"{stats['detectoes_por_minuto']:.1f}/min"),
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
    desenhar_contornos(draw, time.time())

    # ── FOOTER ───────────────────────────────────────────────
    footer_y = H - 72

    draw.rectangle(
        [0, footer_y, W, H],
        fill=(*CORES['BG'], 220)
    )

    draw.line(
        [(0, footer_y), (W, footer_y)],
        fill=(*CORES['BORDA'], 255),
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

        cor_fill  = (*CORES['ACENTO_DIM'], 220) if ativo else (*CORES['CARD_BG'], 200)
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
class Configuracao:
    def __init__(self, arquivo='config.json'):
        self.arquivo = arquivo
        self.config  = {
            "ocr":       {"intervalo": 0.3, "skip_frames": 5,
                          "tamanho_redimensionado": [320, 240], "confianca_minima": 0.7},
            "audio":     {"taxa_fala": 180, "tempo_entre_falas": 3},
            "interface": {"tempo_vida_contorno": 3.0, "mostrar_fps": True},
        }
        try:
            if os.path.exists(arquivo):
                with open(arquivo, 'r', encoding='utf-8') as f:
                    self.config = {**self.config, **json.load(f)}
        except Exception as e:
            logger.warning(f"Erro config: {e}")

    def salvar(self):
        try:
            with open(self.arquivo, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar config: {e}")


class Estatisticas:
    def __init__(self):
        self.valores_detectados = 0
        self.qrcodes_detectados = 0
        self.erros_ocr          = 0
        self.inicio             = time.time()

    def registrar_deteccao(self, tipo):
        if tipo == 'VALOR':    self.valores_detectados += 1
        elif tipo == 'QRCODE': self.qrcodes_detectados += 1

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


class GerenciadorMemoria:
    def __init__(self):
        self.ultima_limpeza = time.time()

    def verificar(self):
        if time.time() - self.ultima_limpeza > 300:
            gc.collect()
            converter_numero_cache.cache_clear()
            self.ultima_limpeza = time.time()
            logger.info("Limpeza de memoria executada")


# ── CONVERSAO NUMERICA ────────────────────────────────────────
@lru_cache(maxsize=100)
def converter_numero_cache(n):
    return _num_pt(n)

def _num_pt(n):
    if n == 0: return "zero"
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
        if n == 100: return "cem"
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
                        tf = f"{converter_numero_cache(n)} reais"

                        if mc:
                            tf += f" e {converter_numero_cache(int(mc.group(1)))} centavos"

                    elif mc:
                        tf = f"{converter_numero_cache(int(mc.group(1)))} centavos"

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

                engine.say(tf)
                engine.runAndWait()

            except Exception as e:
                logger.error(f"Erro na fala: {e}")

    threading.Thread(target=_falar, daemon=True).start()
# ── FILTRO MONETARIO ──────────────────────────────────────────
def validar_valor(val):
    try:
        p = val.split(',')
        if len(p) != 2: return False
        i, d = p[0].replace('.', ''), p[1]
        if len(d) != 2 or not d.isdigit() or not i.isdigit(): return False
        return 0 <= float(f"{i}.{d}") <= 10000
    except:
        return False

def formatar_fala(val):
    p    = val.split(',')
    i, c = int(p[0].replace('.', '')), int(p[1])
    if c == 0: return f"{i} reais"
    if i == 0: return f"{c} centavos"
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

def atualizar_contornos():
    global contornos_ativos
    agora = time.time()
    contornos_ativos = [
        item for item in contornos_ativos
        if (agora - item[2]) < CONTORNO_TEMPO_VIDA
    ]


# ── PROCESSAMENTO ─────────────────────────────────────────────
def processar_valores(frame, estat):
    try:
        small = cv2.resize(frame, RESIZE_OCR)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        res   = reader.readtext(gray, detail=1, paragraph=False,
                                batch_size=1, width_ths=2.0, height_ths=2.0)
        for (bbox, texto, conf) in res:
            if conf > 0.7:
                val = filtrar_valor_monetario(texto)
                if val and evitar_repeticao(val):
                    logger.info(f"Valor: {val} (conf {conf:.2f})")
                    estat.registrar_deteccao('VALOR')
                    historico_valores.append(val)
                    falar_texto(val)
                    sx = 640 / RESIZE_OCR[0]
                    sy = 480 / RESIZE_OCR[1]
                    tl = (int(bbox[0][0] * sx), int(bbox[0][1] * sy))
                    br = (int(bbox[2][0] * sx), int(bbox[2][1] * sy))
                    contornos_ativos.append(((tl, br), f"R$ {val}", time.time(), 'VALOR'))
    except Exception as e:
        logger.error(f"Erro OCR: {e}")
        estat.registrar_erro()

def processar_qrcode(frame, estat):
    data, bbox, _ = detector.detectAndDecode(frame)
    if data and data.strip() and evitar_repeticao(f"QR_{data}"):
        logger.info(f"QR: {data}")
        estat.registrar_deteccao('QRCODE')
        falar_texto("QR Code detectado")
        if bbox is not None:
            pts = bbox.astype(int).reshape(-1, 2)
            tl  = (int(pts[:, 0].min()), int(pts[:, 1].min()))
            br  = (int(pts[:, 0].max()), int(pts[:, 1].max()))
            txt = data[:18] + "..." if len(data) > 18 else data
            contornos_ativos.append(((tl, br), f"QR: {txt}", time.time(), 'QRCODE'))


# ── INICIALIZACAO ─────────────────────────────────────────────
config       = Configuracao()
estatisticas = Estatisticas()
ger_memoria  = GerenciadorMemoria()
configurar_idioma()

OCR_INTERVAL        = config.config['ocr']['intervalo']
SKIP_FRAMES         = config.config['ocr']['skip_frames']
RESIZE_OCR          = tuple(config.config['ocr']['tamanho_redimensionado'])
CONTORNO_TEMPO_VIDA = config.config['interface']['tempo_vida_contorno']

print("[INFO] Aponte a camera para o visor da maquininha ou QR Code...")
logger.info("Sistema PayAI iniciado - 640x480")

# ── LOOP PRINCIPAL ────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        agora = time.time()

        atualizar_contornos()

        if frame_count % 300 == 0:
            ger_memoria.verificar()

        if modo_atual in (MODOS['AUTO'], MODOS['VALORES']):
            if (agora - ultimo_processamento > OCR_INTERVAL and
                    frame_count % SKIP_FRAMES == 0):
                ultimo_processamento = agora
                threading.Thread(target=processar_valores,
                                 args=(frame.copy(), estatisticas),
                                 daemon=True).start()

        if modo_atual in (MODOS['AUTO'], MODOS['QRCODE']):
            processar_qrcode(frame, estatisticas)

        frame_final = desenhar_interface(frame, estatisticas)

        cv2.imshow("PayAI - Sistema Inteligente", frame_final)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            logger.info("Encerrado pelo usuario")
            break
        elif key in (ord('v'), ord('V')):
            modo_atual = MODOS['VALORES']
            falar_texto("Modo valores ativado")
        elif key in (ord('q'), ord('Q')):
            modo_atual = MODOS['QRCODE']
            falar_texto("Modo QR Code ativado")
        elif key in (ord('a'), ord('A')):
            modo_atual = MODOS['AUTO']
            falar_texto("Modo automatico ativado")
        elif key in (ord('i'), ord('I')):
            if idioma_atual == IDIOMAS['PT_BR']:
                idioma_atual = IDIOMAS['ES_CO']
                configurar_idioma()
                falar_texto('Idioma español activado')
            else:
                idioma_atual = IDIOMAS['PT_BR']
                configurar_idioma()
                falar_texto('Idioma portugues ativado')
        elif key in (ord('s'), ord('S')):
            nome = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(nome, frame_final)
            logger.info(f"Screenshot: {nome}")
            falar_texto("Screenshot salvo")
        elif key in (ord('r'), ord('R')):
            if ultima_fala:
                falar_texto(ultima_fala)

except Exception as e:
    logger.error(f"Erro critico: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Estatisticas finais: {estatisticas.obter_estatisticas()}")
    logger.info("Sistema PayAI finalizado")