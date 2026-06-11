import asyncio
import tempfile
import os
import re
import threading
import pygame
from payai.logger import logger
from payai.utils import _num_pt, numero_es
import edge_tts

# inicia mixer na primeira chamada

async def _falar_edge_async(texto, voz):
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


def falar_texto(texto, idioma_atual, ultima_fala_ref):
    def _falar():
        try:
            tf = texto

            mr = re.search(r'(\d+)\s*reais', texto.lower())
            mc = re.search(r'(\d+)\s*centavos', texto.lower())

            # ── PORTUGUES ─────────────────────
            if idioma_atual == 0:  # PT_BR
                if mr:
                    n = int(mr.group(1))
                    tf = f"{_num_pt(n)} reais"
                    if mc:
                        tf += f" e {_num_pt(int(mc.group(1)))} centavos"
                elif mc:
                    tf = f"{_num_pt(int(mc.group(1)))} centavos"

            # ── ESPANHOL ──────────────────────
            else:
                if mr:
                    n = int(mr.group(1))
                    tf = f"{numero_es(n)} pesos colombianos"
                elif mc:
                    tf = f"{numero_es(int(mc.group(1)))} centavos"

                tf = tf.replace('QR Code detectado', 'Código QR detectado')
                tf = tf.replace('Modo automatico ativado', 'Modo automático activado')
                tf = tf.replace('Modo valores ativado', 'Modo valores activado')
                tf = tf.replace('Modo QR Code ativado', 'Modo código QR activado')
                tf = tf.replace('Screenshot salvo', 'Captura guardada')

            if idioma_atual != 0:
                asyncio.run(_falar_edge_async(tf, "es-CO-GonzaloNeural"))
            else:
                asyncio.run(_falar_edge_async(tf, "pt-BR-AntonioNeural"))

        except Exception as e:
            logger.error(f"Erro na fala: {e}")

    threading.Thread(target=_falar, daemon=True).start()
