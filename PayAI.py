import cv2
import easyocr
import pyttsx3
import time
import re
import logging
import threading
from collections import deque
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('payai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializa leitor OCR (português e inglês)
reader = easyocr.Reader(['pt', 'en'], gpu=False)

# Inicializa o mecanismo de voz em português
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Configura para português brasileiro
voices = engine.getProperty('voices')
voz_ptbr = None

for voice in voices:
    if any(term in voice.name.lower() for term in ['portugues', 'brazil', 'brasil', 'pt', 'br']):
        voz_ptbr = voice
        break

if voz_ptbr:
    engine.setProperty('voice', voz_ptbr.id)
    logger.info(f"Voz em português configurada: {voz_ptbr.name}")
else:
    logger.warning("Voz em português não encontrada. Usando voz padrão.")

# Inicializa a câmera com configurações otimizadas
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    logger.error("Erro ao acessar a câmera.")
    exit()

# Inicializa detector de QRCode
detector = cv2.QRCodeDetector()

# Variáveis de controle
ultimo_tempo = time.time()
texto_anterior = ""
historico_valores = deque(maxlen=10)
frame_count = 0
ultimo_processamento = 0

# Para manter os contornos na tela
contornos_ativos = []
CONTORNO_TEMPO_VIDA = 3.0

# Configurações de performance
OCR_INTERVAL = 0.3
SKIP_FRAMES = 5
RESIZE_OCR = (320, 240)

# Thread-safe para fala
fala_lock = threading.Lock()

# Sistema de modos
MODOS = {
    'AUTO': 0,      # Detecta automaticamente QR Code e Valores
    'VALORES': 1,   # Foca apenas em valores monetários
    'QRCODE': 2     # Foca apenas em QR Codes
}
modo_atual = MODOS['AUTO']

# Cores e estilo
CORES = {
    'VALOR': (0, 255, 0),       # Verde para valores
    'QRCODE': (255, 0, 255),    # Magenta para QR Codes
    'MODO': (0, 255, 255),      # Amarelo para informações de modo
    'BACKGROUND': (0, 0, 0),    # Preto para fundo
    'TEXTO': (255, 255, 255)    # Branco para texto
}

def converter_numero_para_portugues(numero):
    """Converte números para palavras em português de forma natural"""
    if numero == 0:
        return "zero"
    
    # Para números pequenos (até 99), converte completamente
    if numero <= 99:
        unidades = ['', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
        dez_a_dezenove = ['dez', 'onze', 'doze', 'treze', 'quatorze', 'quinze', 'dezesseis', 'dezessete', 'dezoito', 'dezenove']
        dezenas = ['', '', 'vinte', 'trinta', 'quarenta', 'cinquenta', 'sessenta', 'setenta', 'oitenta', 'noventa']
        
        if 1 <= numero <= 9:
            return unidades[numero]
        elif 10 <= numero <= 19:
            return dez_a_dezenove[numero - 10]
        else:
            dezena = numero // 10
            unidade = numero % 10
            if unidade == 0:
                return dezenas[dezena]
            else:
                return f"{dezenas[dezena]} e {unidades[unidade]}"
    
    # Para números maiores (100+), mantém a lógica anterior simplificada
    elif numero <= 999:
        centenas = ['', 'cento', 'duzentos', 'trezentos', 'quatrocentos', 'quinhentos', 'seiscentos', 'setecentos', 'oitocentos', 'novecentos']
        
        if numero == 100:
            return "cem"
        
        texto = ""
        centena = numero // 100
        resto = numero % 100
        
        if centena > 0:
            texto += centenas[centena]
            if resto > 0:
                texto += " e "
        
        if resto > 0:
            texto += converter_numero_para_portugues(resto)
        
        return texto
    
    else:
        # Para números muito grandes, formata com pontos
        return f"{numero:,}".replace(",", ".")

def falar_texto(texto):
    """Função thread-safe para falar em português"""
    def falar():
        with fala_lock:
            try:
                # Se for um valor monetário, converte o número
                if 'reais' in texto.lower() or 'centavos' in texto.lower():
                    # Extrai números do texto
                    if ' e ' in texto:
                        # Caso "X reais e Y centavos"
                        partes = texto.split(' e ')
                        if len(partes) == 2:
                            reais_match = re.search(r'(\d+)\s*reais', partes[0])
                            centavos_match = re.search(r'(\d+)\s*centavos', partes[1])
                            
                            if reais_match and centavos_match:
                                reais_num = int(reais_match.group(1))
                                centavos_num = int(centavos_match.group(1))
                                
                                if reais_num <= 9999:
                                    reais_ptbr = converter_numero_para_portugues(reais_num)
                                    centavos_ptbr = converter_numero_para_portugues(centavos_num)
                                    texto_final = f"{reais_ptbr} reais e {centavos_ptbr} centavos"
                                else:
                                    texto_final = texto
                            else:
                                texto_final = texto
                        else:
                            texto_final = texto
                    else:
                        # Caso simples "X reais" ou "Y centavos"
                        match_reais = re.search(r'(\d+)\s*reais', texto.lower())
                        match_centavos = re.search(r'(\d+)\s*centavos', texto.lower())
                        
                        if match_reais:
                            numero = int(match_reais.group(1))
                            if numero <= 9999:
                                numero_ptbr = converter_numero_para_portugues(numero)
                                texto_final = f"{numero_ptbr} reais"
                            else:
                                texto_final = f"{numero:,} reais".replace(",", ".")
                        elif match_centavos:
                            numero = int(match_centavos.group(1))
                            numero_ptbr = converter_numero_para_portugues(numero)
                            texto_final = f"{numero_ptbr} centavos"
                        else:
                            texto_final = texto
                else:
                    texto_final = texto
                
                engine.say(texto_final)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Erro na fala: {e}")
    
    threading.Thread(target=falar, daemon=True).start()

def filtrar_valor_monetario(texto):
    """Filtra e formata valores monetários - identifica vírgula corretamente"""
    # Padrões mais flexíveis para capturar valores como "15,00"
    padroes = [
        r'(\d{1,3}(?:\.\d{3})*,\d{2})',  # 1.500,00 ou 15,00
        r'(\d+,\d{2})',                   # 15,00
        r'R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',  # R$ 15,00
        r'R\$\s*(\d+,\d{2})',                   # R$ 15,00
    ]
    
    for padrao in padroes:
        match = re.search(padrao, texto)
        if match:
            valor_com_virgula = match.group(1)
            
            # Verifica se tem pelo menos 1 dígito antes da vírgula
            partes = valor_com_virgula.split(',')
            if len(partes) == 2 and partes[0].isdigit():
                # Remove pontos de milhar se existirem
                parte_inteira = partes[0].replace('.', '')
                parte_decimal = partes[1]
                
                # Converte para número
                try:
                    valor_numerico = float(f"{parte_inteira}.{parte_decimal}")
                    
                    # Se os centavos são 00, fala como inteiro
                    if parte_decimal == '00':
                        valor_int = int(parte_inteira)
                        return f"{valor_int} reais"
                    else:
                        # Para valores com centavos, fala "X reais e Y centavos"
                        inteiro = int(parte_inteira)
                        centavos = int(parte_decimal)
                        
                        if inteiro == 0:
                            return f"{centavos} centavos"
                        elif centavos == 0:
                            return f"{inteiro} reais"
                        else:
                            return f"{inteiro} reais e {centavos} centavos"
                            
                except ValueError:
                    continue
    
    return None

def evitar_repeticao(texto, tempo_minimo=3):
    """Evita leituras repetidas de forma mais eficiente"""
    global ultimo_tempo, texto_anterior
    agora = time.time()
    
    if texto == texto_anterior and (agora - ultimo_tempo) < tempo_minimo:
        return False
    
    texto_anterior = texto
    ultimo_tempo = agora
    return True

def atualizar_contornos_ativos():
    """Remove contornos antigos"""
    global contornos_ativos
    agora = time.time()
    contornos_ativos = [(bbox, texto, ts, tipo) for bbox, texto, ts, tipo in contornos_ativos 
                       if (agora - ts) < CONTORNO_TEMPO_VIDA]

def desenhar_texto_bonito(frame, texto, posicao, cor, tamanho=0.7, espessura=2):
    """Desenha texto com sombra para melhor legibilidade"""
    x, y = posicao
    
    # Sombra
    cv2.putText(frame, texto, (x+2, y+2), 
                cv2.FONT_HERSHEY_SIMPLEX, tamanho, CORES['BACKGROUND'], espessura+1)
    
    # Texto principal
    cv2.putText(frame, texto, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, tamanho, cor, espessura)

def desenhar_interface(frame):
    """Desenha a interface do usuário"""
    altura = frame.shape[0]
    largura = frame.shape[1]
    
    # Barra superior
    cv2.rectangle(frame, (0, 0), (largura, 40), CORES['BACKGROUND'], -1)
    
    # Título
    desenhar_texto_bonito(frame, "PayAI - Sistema de Reconhecimento", (10, 30), CORES['TEXTO'], 0.8, 1)
    
    # Modo atual
    modos_texto = ["AUTO", "VALORES", "QR CODE"]
    modo_texto = modos_texto[modo_atual]
    desenhar_texto_bonito(frame, f"Modo: {modo_texto}", (largura - 200, 30), CORES['MODO'], 0.7, 1)
    
    # Barra inferior com instruções
    cv2.rectangle(frame, (0, altura-60), (largura, altura), CORES['BACKGROUND'], -1)
    
    instrucoes = [
        "ESC: Sair",
        "V: Modo Valores", 
        "Q: Modo QR Code",
        "A: Modo Automático"
    ]
    
    for i, instrucao in enumerate(instrucoes):
        desenhar_texto_bonito(frame, instrucao, (10 + i*150, altura-20), CORES['TEXTO'], 0.5, 1)

def processar_valores(frame):
    """Processa valores monetários"""
    try:
        small_frame = cv2.resize(frame, RESIZE_OCR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        results = reader.readtext(
            gray, 
            detail=1, 
            paragraph=False, 
            batch_size=1,
            width_ths=2.0,
            height_ths=2.0
        )
        
        for (bbox, texto, conf) in results:
            if conf > 0.7:
                valor_monetario = filtrar_valor_monetario(texto)
                
                if valor_monetario and evitar_repeticao(valor_monetario):
                    logger.info(f"Valor detectado: {valor_monetario} (confiança: {conf:.2f})")
                    falar_texto(valor_monetario)
                    
                    # Ajusta coordenadas para frame original
                    scale_x = 640 / RESIZE_OCR[0]
                    scale_y = 480 / RESIZE_OCR[1]
                    top_left = (int(bbox[0][0] * scale_x), int(bbox[0][1] * scale_y))
                    bottom_right = (int(bbox[2][0] * scale_x), int(bbox[2][1] * scale_y))
                    
                    # Adiciona à lista de contornos ativos
                    contornos_ativos.append((
                        (top_left, bottom_right), 
                        valor_monetario, 
                        time.time(),
                        'VALOR'
                    ))
    except Exception as e:
        logger.error(f"Erro no OCR: {e}")

def processar_qrcode(frame):
    """Processa QR Codes"""
    data, bbox, _ = detector.detectAndDecode(frame)
    if data and data.strip():
        if evitar_repeticao(f"QR_{data}"):
            logger.info(f"QR Code detectado: {data}")
            falar_texto("QR Code detectado")
            
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for j in range(len(pts)):
                    cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1) % len(pts)]), CORES['QRCODE'], 3)
                
                # Adiciona contorno do QR Code
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                top_left = (min(x_coords), min(y_coords))
                bottom_right = (max(x_coords), max(y_coords))
                
                contornos_ativos.append((
                    (top_left, bottom_right), 
                    f"QR: {data[:20]}...", 
                    time.time(),
                    'QRCODE'
                ))

print("[INFO] Aponte a câmera para o visor da maquininha ou QR Code...")
logger.info("Sistema PayAI iniciado - Interface Melhorada + Tri-Mode")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        agora = time.time()

        # Atualiza contornos (remove os antigos)
        atualizar_contornos_ativos()

        # --- PROCESSAMENTO BASEADO NO MODO ---
        if modo_atual == MODOS['AUTO'] or modo_atual == MODOS['VALORES']:
            if (agora - ultimo_processamento > OCR_INTERVAL and 
                frame_count % SKIP_FRAMES == 0):
                
                ultimo_processamento = agora
                threading.Thread(target=processar_valores, args=(frame.copy(),), daemon=True).start()

        if modo_atual == MODOS['AUTO'] or modo_atual == MODOS['QRCODE']:
            processar_qrcode(frame)

        # --- DESENHA CONTORNOS ATIVOS ---
        for (top_left, bottom_right), texto, timestamp, tipo in contornos_ativos:
            tempo_restante = CONTORNO_TEMPO_VIDA - (agora - timestamp)
            alpha = max(0.4, tempo_restante / CONTORNO_TEMPO_VIDA)
            
            # Escolhe cor baseada no tipo
            cor = CORES['VALOR'] if tipo == 'VALOR' else CORES['QRCODE']
            
            # Desenha retângulo semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, cor, 2)
            desenhar_texto_bonito(overlay, texto, (top_left[0], top_left[1]-15), cor, 0.6, 1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- DESENHA INTERFACE ---
        desenhar_interface(frame)

        # Exibe o vídeo
        cv2.imshow("PayAI - Sistema Avançado", frame)

        # --- CONTROLE DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - Sair
            logger.info("Sistema encerrado pelo usuário")
            break
        elif key == ord('v') or key == ord('V'):  # V - Modo Valores
            modo_atual = MODOS['VALORES']
            falar_texto("Modo valores ativado")
            logger.info("Modo alterado para VALORES")
        elif key == ord('q') or key == ord('Q'):  # Q - Modo QR Code
            modo_atual = MODOS['QRCODE']
            falar_texto("Modo QR Code ativado")
            logger.info("Modo alterado para QRCODE")
        elif key == ord('a') or key == ord('A'):  # A - Modo Automático
            modo_atual = MODOS['AUTO']
            falar_texto("Modo automático ativado")
            logger.info("Modo alterado para AUTO")

except Exception as e:
    logger.error(f"Erro crítico no sistema: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Sistema PayAI finalizado")