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
import payai.ocr as ocr_mod
from payai.ocr import OCRThread
from payai.speech import falar_texto

global ultimo_fps_tempo
global fps_atual

# ── LOGGING ───────────────────────────────────────────────────
# logging agora gerenciado por payai.logger
# logger importado do pacote payai

# ── OCR / VOZ ─────────────────────────────────────────────────
# ocr será gerenciado pelo módulo payai.ocr (ocr_mod)

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

# Estatisticas (classe simples usada pelo sistema)
class Estatisticas:
    def __init__(self):
        self.valores_detectados = 0
        self.qrcodes_detectados = 0
        self.erros_ocr = 0
        self.inicio = time.time()

    def registrar_deteccao(self, tipo):
        if tipo == 'VALOR':
            self.valores_detectados += 1
        elif tipo == 'QRCODE':
            self.qrcodes_detectados += 1

    def registrar_erro(self):
        self.erros_ocr += 1

    def obter_estatisticas(self):
        t = time.time() - self.inicio
        total = self.valores_detectados + self.qrcodes_detectados
        return {
            'tempo_execucao': t,
            'valores_detectados': self.valores_detectados,
            'qrcodes_detectados': self.qrcodes_detectados,
            'erros_ocr': self.erros_ocr,
            'detectoes_por_minuto': total / (t / 60) if t > 0 else 0,
        }

# instancia global
estatisticas = Estatisticas()

fps_atual = 0
ultimo_fps_tempo = time.time()
tempo_ocr = 0
regioes_detectadas = 0

fila_ocr = Queue(maxsize=OCR_QUEUE_SIZE)

fala_lock            = threading.Lock()
ultima_fala          = ""

# Estado OCR (delegado ao módulo ocr_mod)
# ocr_rodando = False

modo_atual = MODOS['AUTO']
idioma_atual = IDIOMAS['PT_BR']
# CORES, MODOS e IDIOMAS já são importados de payai.config

# Último frame exibido (para screenshots)
last_frame = None

# ── FONTES ───────────────────────────────────────────────────
# FONTES já carregadas pelo módulo payai.draw

# ── UTILITARIOS DE DESENHO ───────────────────────────────────
def cv2_para_pil(frame):
    # compatibilidade com o módulo draw: converte BGR->RGB
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_para_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── CARREGAMENTO OCR ──────────────────────────────────────────
# OCR é carregado pelo módulo payai.ocr (ocr_mod.carregar_ocr)

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

    # Estatísticas
    draw.text(
        (W - 14 - 120, 14),
        f"FPS: {fps_atual:.1f}",
        font=FONTES['subtitulo'],
        fill=CORES['TEXTO_SEC']
    )

    # ── BODY ────────────────────────────────────────────────────
    modos_map = {
        MODOS['AUTO']: 'AUTO',
        MODOS['VALORES']: 'VALORES',
        MODOS['QRCODE']: 'QR CODE'
    }
    modo_label = modos_map.get(modo_atual, 'AUTO')
    draw.text(
        (14, 50),
        modo_label,
        font=FONTES['modo'],
        fill=CORES['TEXTO_PRIM']
    )

    # Idioma atual
    idioma_label = 'PT-BR' if idioma_atual == IDIOMAS['PT_BR'] else 'ES-CO'
    dx = FONTES['modo'].getlength(idioma_label)
    draw.text(
        (W - 14 - dx, 50),
        f"Idioma: {idioma_label}",
        font=FONTES['subtitulo'],
        fill=CORES['TEXTO_SEC']
    )

    # Estatísticas detalhadas (debug)
    if estatisticas:
        linha = 70
        for chave, valor in estatisticas.items():
            draw.text(
                (14, linha),
                f"{chave}: {valor}",
                font=FONTES['corpo'],
                fill=CORES['TEXTO_PRIM']
            )
            linha += 20

    # ── CONTORNOS ───────────────────────────────────────────
    desenhar_contornos(draw, agora)

    # ── FOOTER ─────────────────────────────────────────────────
    draw.rectangle([0, H - 42, W, H], fill=CORES['BG'])
    draw.line(
        [(0, H - 42), (W, H - 42)], 
        fill=CORES['BORDA'],
        width=1
    ) 

    draw.text(
        (14, H - 28),
        "Desenvolvido por PayAI",
        font=FONTES['subtitulo'],
        fill=CORES['TEXTO_SEC']
    )

    return pil_para_cv2(img)


# ── Teclas / Handler ─────────────────────────────────────────
async def tratar_tecla(tecla):
    """Trata teclas pressionadas. Recebe o valor retornado por cv2.waitKey() (normalizado com &0xFF)."""
    global modo_atual, idioma_atual, ultima_fala, last_frame

    try:
        if tecla == 27:  # ESC
            logger.info("Encerrado pelo usuario")
            raise SystemExit

        elif tecla in (ord('v'), ord('V')):
            modo_atual = MODOS['VALORES']
            falar_texto("Modo valores ativado", idioma_atual, None)

        elif tecla in (ord('q'), ord('Q')):
            modo_atual = MODOS['QRCODE']
            falar_texto("Modo QR Code ativado", idioma_atual, None)

        elif tecla in (ord('a'), ord('A')):
            modo_atual = MODOS['AUTO']
            falar_texto("Modo automatico ativado", idioma_atual, None)

        elif tecla in (ord('i'), ord('I')):
            if idioma_atual == IDIOMAS['PT_BR']:
                idioma_atual = IDIOMAS['ES_CO']
                falar_texto('Idioma español activado', idioma_atual, None)
            else:
                idioma_atual = IDIOMAS['PT_BR']
                falar_texto('Idioma portugues ativado', idioma_atual, None)

        elif tecla in (ord('s'), ord('S')):
            if last_frame is not None:
                nome = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                try:
                    cv2.imwrite(nome, last_frame)
                    logger.info(f"Screenshot: {nome}")
                    falar_texto("Screenshot salvo", idioma_atual, None)
                except Exception as e:
                    logger.error(f"Erro ao salvar screenshot: {e}")

        elif tecla in (ord('r'), ord('R')):
            if ultima_fala:
                falar_texto(ultima_fala, idioma_atual, None)

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Erro ao tratar tecla: {e}")


# ── PROCESSAMENTO OCR (wrapper) ───────────────────────────────
def processar_ocr(frame):
    """Wrapper para rodar OCR em background e acionar TTS quando houver valor detectado."""
    global ultima_fala
    try:
        val = ocr_mod.processar_valores(frame, estatisticas)
        if val:
            ultima_fala = val
            falar_texto(val, idioma_atual, None)
    except Exception as e:
        logger.error(f"Erro processar_ocr: {e}")


# ── LOOP PRINCIPAL ─────────────────────────────────────────────
async def loop_principal():
    global ultimo_tempo, frame_count, ultimo_processamento, regioes_detectadas
    global ocr_rodando, progresso_loading, ultimo_fps_tempo, fps_atual, last_frame

    while True:
        agora = time.time()

        # Atualiza FPS a cada segundo
        if agora - ultimo_fps_tempo >= 1:
            fps_atual = frame_count / (agora - ultimo_fps_tempo) if (agora - ultimo_fps_tempo) > 0 else 0
            ultimo_fps_tempo = agora

        # Captura de frame
        if camera_pronta and cap is not None:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # garante resolução esperada
            try:
                frame = cv2.resize(frame, (LARGURA, ALTURA))
            except Exception:
                pass

            frame_count += 1

            # Dispara OCR em background quando pronto
            if getattr(ocr_mod, 'ocr_pronto', False) and not getattr(ocr_mod, 'ocr_rodando', False):
                threading.Thread(target=processar_ocr, args=(frame.copy(),), daemon=True).start()

            # Desenha interface (contornos são desenhados dentro da interface)
            try:
                frame_final = desenhar_interface(frame, estatisticas.obter_estatisticas(), agora)
            except Exception as e:
                logger.error(f"Erro ao desenhar interface: {e}")
                frame_final = frame

            # Atualiza last_frame para permitir screenshot
            try:
                last_frame = frame_final.copy()
            except Exception:
                last_frame = frame_final

            cv2.imshow('PayAI', frame_final)

        else:
            # Splash / loading simples enquanto a camera ou OCR iniciam
            splash = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
            splash[:] = CORES['BG']
            try:
                img = cv2_para_pil(splash)
                draw = ImageDraw.Draw(img)
                texto_c(draw, "PAYAI", LARGURA // 2, 165, FONTES['logo_pay'], CORES['ACENTO'])
                texto_c(draw, "Inicializando...", LARGURA // 2, 225, FONTES['corpo_b'], CORES['TEXTO_PRIM'])
                splash_img = pil_para_cv2(img)
            except Exception:
                splash_img = splash

            cv2.imshow('PayAI', splash_img)

        # leitura de teclas
        key = cv2.waitKey(1)
        if key != -1 and key is not None:
            try:
                await tratar_tecla(key & 0xFF)
            except SystemExit:
                raise
            except Exception as e:
                logger.error(f"Erro no handler de tecla: {e}")

        await asyncio.sleep(0.01)

# ── INICIO DO PROGRAMA ────────────────────────────────────────
if __name__ == '__main__':

    # permitir execução direta
    print('[INFO] Executando main.py')

    # Carregar recursos
    # Carrega OCR em background via modulo
    threading.Thread(target=ocr_mod.carregar_ocr, daemon=True).start()

    # Fontes (já carregadas no import) — opcional chamar novamente
    try:
        carregar_fontes()
    except Exception:
        pass

    # Iniciar loop principal
    try:
        asyncio.run(loop_principal())
    except SystemExit:
        logger.info('Encerrando por solicitação do usuário')
    except Exception as e:
        logger.error(f'Erro fatal: {e}')

    # Limpeza
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    cv2.destroyAllWindows()
