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

# Paleta de cores branco, preto e cinza
CORES = {
    'BRANCO': (255, 255, 255),      # Branco puro
    'PRETO': (0, 0, 0),            # Preto puro
    'CINZA_ESCURO': (40, 40, 40),  # Cinza escuro
    'CINZA_MEDIO': (100, 100, 100), # Cinza médio
    'CINZA_CLARO': (180, 180, 180), # Cinza claro
    'CINZA_CARDBG': (50, 50, 50),   # Cinza para cards
    'DESTAQUE': (200, 200, 200),    # Cinza de destaque
    'SUCESSO': (150, 150, 150),     # Cinza para sucesso
    'ALERTA': (120, 120, 120)       # Cinza para alertas
}

class Configuracao:
    def __init__(self, arquivo_config='config.json'):
        self.arquivo_config = arquivo_config
        self.config = self.carregar_config()
    
    def carregar_config(self):
        """Carrega configurações do arquivo JSON"""
        config_padrao = {
            "ocr": {
                "intervalo": 0.3,
                "skip_frames": 5,
                "tamanho_redimensionado": [320, 240],
                "confianca_minima": 0.7
            },
            "audio": {
                "taxa_fala": 180,
                "tempo_entre_falas": 3
            },
            "interface": {
                "tempo_vida_contorno": 3.0,
                "mostrar_fps": True
            }
        }
        
        try:
            if os.path.exists(self.arquivo_config):
                with open(self.arquivo_config, 'r', encoding='utf-8') as f:
                    return {**config_padrao, **json.load(f)}
        except Exception as e:
            logger.warning(f"Erro ao carregar config: {e}")
        
        return config_padrao
    
    def salvar_config(self):
        """Salva configurações atualizadas"""
        try:
            with open(self.arquivo_config, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar config: {e}")

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
        tempo_execucao = time.time() - self.inicio
        total_detectoes = self.valores_detectados + self.qrcodes_detectados
        detectoes_por_minuto = total_detectoes / (tempo_execucao / 60) if tempo_execucao > 0 else 0
        
        return {
            'tempo_execucao': tempo_execucao,
            'valores_detectados': self.valores_detectados,
            'qrcodes_detectados': self.qrcodes_detectados,
            'erros_ocr': self.erros_ocr,
            'detectoes_por_minuto': detectoes_por_minuto
        }

class GerenciadorMemoria:
    def __init__(self, limiar_memoria=80, intervalo_limpeza=300):
        self.limiar_memoria = limiar_memoria
        self.ultima_limpeza = time.time()
        self.intervalo_limpeza = intervalo_limpeza
    
    def verificar_memoria(self):
        """Verifica e gerencia uso de memória"""
        agora = time.time()
        
        if agora - self.ultima_limpeza > self.intervalo_limpeza:
            # Forçar coleta de lixo
            gc.collect()
            
            # Limpar cache se necessário
            converter_numero_cache.cache_clear()
            
            self.ultima_limpeza = agora
            logger.info("Limpeza de memória executada")

@lru_cache(maxsize=100)
def converter_numero_cache(numero):
    """Versão em cache da conversão de números"""
    return converter_numero_para_portugues(numero)

def converter_numero_para_portugues(numero):
    """Converte números para palavras em português de forma natural"""
    if numero == 0:
        return "zero"
    
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
        return f"{numero:,}".replace(",", ".")

def validar_valor_monetario(valor_str):
    """Valida se é um valor monetário plausível"""
    try:
        partes = valor_str.split(',')
        if len(partes) != 2:
            return False
        
        inteiro = partes[0].replace('.', '')
        decimal = partes[1]
        
        # Verificar se decimal tem 2 dígitos
        if len(decimal) != 2 or not decimal.isdigit():
            return False
            
        # Verificar se a parte inteira é numérica
        if not inteiro.isdigit():
            return False
            
        valor = float(f"{inteiro}.{decimal}")
        
        # Valores monetários razoáveis (até R$ 10.000,00)
        return 0 <= valor <= 10000.00
        
    except (ValueError, IndexError):
        return False

def formatar_valor_para_fala(valor_str):
    """Formata o valor monetário para fala"""
    partes = valor_str.split(',')
    parte_inteira = partes[0].replace('.', '')
    parte_decimal = partes[1]
    
    try:
        valor_numerico = float(f"{parte_inteira}.{parte_decimal}")
        
        if parte_decimal == '00':
            valor_int = int(parte_inteira)
            return f"{valor_int} reais"
        else:
            inteiro = int(parte_inteira)
            centavos = int(parte_decimal)
            
            if inteiro == 0:
                return f"{centavos} centavos"
            elif centavos == 0:
                return f"{inteiro} reais"
            else:
                return f"{inteiro} reais e {centavos} centavos"
                
    except ValueError:
        return None

def melhorar_filtro_valor_monetario(texto):
    """Filtro mais robusto para valores monetários"""
    # Padrões mais abrangentes
    padroes = [
        r'R\s*[\$\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*R',
        r'valor\s*[\:\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'total\s*[\:\s]*\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d+,\d{2})\s*(?:reais|R\$)',
    ]
    
    texto_limpo = re.sub(r'\s+', ' ', texto.strip())
    
    for padrao in padroes:
        match = re.search(padrao, texto_limpo, re.IGNORECASE)
        if match:
            valor_str = match.group(1)
            # Validação adicional
            if validar_valor_monetario(valor_str):
                return formatar_valor_para_fala(valor_str)
    
    return None

def falar_texto(texto):
    """Função thread-safe para falar em português"""
    def falar():
        with fala_lock:
            try:
                if 'reais' in texto.lower() or 'centavos' in texto.lower():
                    if ' e ' in texto:
                        partes = texto.split(' e ')
                        if len(partes) == 2:
                            reais_match = re.search(r'(\d+)\s*reais', partes[0])
                            centavos_match = re.search(r'(\d+)\s*centavos', partes[1])
                            
                            if reais_match and centavos_match:
                                reais_num = int(reais_match.group(1))
                                centavos_num = int(centavos_match.group(1))
                                
                                if reais_num <= 9999:
                                    reais_ptbr = converter_numero_cache(reais_num)
                                    centavos_ptbr = converter_numero_cache(centavos_num)
                                    texto_final = f"{reais_ptbr} reais e {centavos_ptbr} centavos"
                                else:
                                    texto_final = texto
                            else:
                                texto_final = texto
                        else:
                            texto_final = texto
                    else:
                        match_reais = re.search(r'(\d+)\s*reais', texto.lower())
                        match_centavos = re.search(r'(\d+)\s*centavos', texto.lower())
                        
                        if match_reais:
                            numero = int(match_reais.group(1))
                            if numero <= 9999:
                                numero_ptbr = converter_numero_cache(numero)
                                texto_final = f"{numero_ptbr} reais"
                            else:
                                texto_final = f"{numero:,} reais".replace(",", ".")
                        elif match_centavos:
                            numero = int(match_centavos.group(1))
                            numero_ptbr = converter_numero_cache(numero)
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
    # Primeiro tenta o filtro melhorado
    valor_melhorado = melhorar_filtro_valor_monetario(texto)
    if valor_melhorado:
        return valor_melhorado
    
    # Fallback para o filtro original
    padroes = [
        r'(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'(\d+,\d{2})',
        r'R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})',
        r'R\$\s*(\d+,\d{2})',
    ]
    
    for padrao in padroes:
        match = re.search(padrao, texto)
        if match:
            valor_com_virgula = match.group(1)
            partes = valor_com_virgula.split(',')
            if len(partes) == 2 and partes[0].isdigit():
                parte_inteira = partes[0].replace('.', '')
                parte_decimal = partes[1]
                
                try:
                    valor_numerico = float(f"{parte_inteira}.{parte_decimal}")
                    
                    if parte_decimal == '00':
                        valor_int = int(parte_inteira)
                        return f"{valor_int} reais"
                    else:
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

def desenhar_texto_estilizado(frame, texto, posicao, cor, tamanho=0.7, espessura=2, sombra=True):
    """Desenha texto com estilo moderno"""
    x, y = posicao
    
    if sombra:
        # Sombra preta
        cv2.putText(frame, texto, (x+1, y+1), 
                    cv2.FONT_HERSHEY_SIMPLEX, tamanho, CORES['PRETO'], espessura + 2)
    
    # Texto principal
    cv2.putText(frame, texto, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, tamanho, cor, espessura)

def criar_card_arredondado(frame, x1, y1, x2, y2, cor, alpha=0.9):
    """Cria um card com cantos arredondados"""
    overlay = frame.copy()
    
    # Retângulo principal
    cv2.rectangle(overlay, (x1, y1), (x2, y2), cor, -1)
    
    # Aplica transparência
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Borda sutil
    cv2.rectangle(frame, (x1, y1), (x2, y2), CORES['CINZA_CLARO'], 1)

def desenhar_estatisticas(frame, estatisticas):
    """Desenha estatísticas na tela"""
    stats = estatisticas.obter_estatisticas()
    texto_stats = [
        f"Valores: {stats['valores_detectados']}",
        f"QR Codes: {stats['qrcodes_detectados']}",
        f"Taxa: {stats['detectoes_por_minuto']:.1f}/min"
    ]
    
    for i, texto in enumerate(texto_stats):
        y_pos = 80 + i * 25
        desenhar_texto_estilizado(frame, texto, (20, y_pos), CORES['CINZA_CLARO'], 0.5, 1)

def desenhar_interface_monocromatica(frame, estatisticas=None, config=None):
    """Desenha interface em preto, branco e cinza"""
    altura = frame.shape[0]
    largura = frame.shape[1]
    
    # Header em cinza escuro
    cv2.rectangle(frame, (0, 0), (largura, 50), CORES['CINZA_ESCURO'], -1)
    
    # Logo/Título
    desenhar_texto_estilizado(frame, "PAY AI", (20, 35), CORES['BRANCO'], 1.1, 2)
    desenhar_texto_estilizado(frame, "Sistema Inteligente", (150, 35), CORES['CINZA_CLARO'], 0.6, 1)
    
    # Card do modo atual
    modos_texto = ["AUTO", "VALORES", "QR CODE"]
    modo_texto = modos_texto[modo_atual]
    criar_card_arredondado(frame, largura - 180, 15, largura - 10, 45, CORES['CINZA_CARDBG'])
    desenhar_texto_estilizado(frame, f"MODO: {modo_texto}", (largura - 170, 35), CORES['DESTAQUE'], 0.6, 1)
    
    # Footer em cinza escuro
    footer_y = altura - 70
    cv2.rectangle(frame, (0, footer_y), (largura, altura), CORES['CINZA_ESCURO'], -1)
    
    # Instruções em cards individuais
    instrucoes = [
        ("ESC", "SAIR"),
        ("V", "VALORES"), 
        ("Q", "QR CODE"),
        ("A", "AUTO")
    ]
    
    card_largura = 120
    espacamento = 10
    inicio_x = (largura - (len(instrucoes) * card_largura + (len(instrucoes)-1) * espacamento)) // 2
    
    for i, (tecla, descricao) in enumerate(instrucoes):
        x1 = inicio_x + i * (card_largura + espacamento)
        x2 = x1 + card_largura
        y1 = footer_y + 10
        y2 = altura - 10
        
        criar_card_arredondado(frame, x1, y1, x2, y2, CORES['CINZA_CARDBG'])
        
        # Tecla em branco
        desenhar_texto_estilizado(frame, tecla, (x1 + 45, y1 + 25), CORES['BRANCO'], 0.9, 2)
        # Descrição em cinza claro
        desenhar_texto_estilizado(frame, descricao, (x1 + 25, y1 + 45), CORES['CINZA_CLARO'], 0.5, 1)
    
    # Estatísticas se configurado
    if config and config.config['interface']['mostrar_fps'] and estatisticas:
        desenhar_estatisticas(frame, estatisticas)

def processar_valores(frame, estatisticas):
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
                    estatisticas.registrar_deteccao('VALOR')
                    falar_texto(valor_monetario)
                    
                    scale_x = 640 / RESIZE_OCR[0]
                    scale_y = 480 / RESIZE_OCR[1]
                    top_left = (int(bbox[0][0] * scale_x), int(bbox[0][1] * scale_y))
                    bottom_right = (int(bbox[2][0] * scale_x), int(bbox[2][1] * scale_y))
                    
                    contornos_ativos.append((
                        (top_left, bottom_right), 
                        f"VALOR: {valor_monetario}", 
                        time.time(),
                        'VALOR'
                    ))
    except Exception as e:
        logger.error(f"Erro no OCR: {e}")
        estatisticas.registrar_erro()

def processar_qrcode(frame, estatisticas):
    """Processa QR Codes"""
    data, bbox, _ = detector.detectAndDecode(frame)
    if data and data.strip():
        if evitar_repeticao(f"QR_{data}"):
            logger.info(f"QR Code detectado: {data}")
            estatisticas.registrar_deteccao('QRCODE')
            falar_texto("QR Code detectado")
            
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for j in range(len(pts)):
                    cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1) % len(pts)]), CORES['BRANCO'], 3)
                
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                top_left = (min(x_coords), min(y_coords))
                bottom_right = (max(x_coords), max(y_coords))
                
                texto_qr = data[:15] + "..." if len(data) > 15 else data
                contornos_ativos.append((
                    (top_left, bottom_right), 
                    f"QR: {texto_qr}", 
                    time.time(),
                    'QRCODE'
                ))

# Inicializar componentes
config = Configuracao()
estatisticas = Estatisticas()
gerenciador_memoria = GerenciadorMemoria()

# Atualizar constantes com configuração
OCR_INTERVAL = config.config['ocr']['intervalo']
SKIP_FRAMES = config.config['ocr']['skip_frames']
RESIZE_OCR = tuple(config.config['ocr']['tamanho_redimensionado'])
CONTORNO_TEMPO_VIDA = config.config['interface']['tempo_vida_contorno']

print("[INFO] Aponte a câmera para o visor da maquininha ou QR Code...")
logger.info("Sistema PayAI iniciado - Interface Monocromática")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        agora = time.time()

        # Atualiza contornos
        atualizar_contornos_ativos()

        # Verificação de memória periódica
        if frame_count % 300 == 0:  # A cada ~10 segundos
            gerenciador_memoria.verificar_memoria()

        # --- PROCESSAMENTO BASEADO NO MODO ---
        if modo_atual == MODOS['AUTO'] or modo_atual == MODOS['VALORES']:
            if (agora - ultimo_processamento > OCR_INTERVAL and 
                frame_count % SKIP_FRAMES == 0):
                
                ultimo_processamento = agora
                threading.Thread(target=processar_valores, args=(frame.copy(), estatisticas), daemon=True).start()

        if modo_atual == MODOS['AUTO'] or modo_atual == MODOS['QRCODE']:
            processar_qrcode(frame, estatisticas)

        # --- DESENHA CONTORNOS ATIVOS ---
        for (top_left, bottom_right), texto, timestamp, tipo in contornos_ativos:
            tempo_restante = CONTORNO_TEMPO_VIDA - (agora - timestamp)
            alpha = max(0.5, tempo_restante / CONTORNO_TEMPO_VIDA)
            
            # Usa branco para ambos os tipos (valores e QR codes)
            cor = CORES['BRANCO']
            
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, cor, 3)
            cv2.putText(overlay, texto, (top_left[0], top_left[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- DESENHA INTERFACE MONOCROMÁTICA ---
        desenhar_interface_monocromatica(frame, estatisticas, config)

        # Exibe o vídeo
        cv2.imshow("PayAI - Sistema Inteligente", frame)

        # --- CONTROLE DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            logger.info("Sistema encerrado pelo usuário")
            break
        elif key == ord('v') or key == ord('V'):
            modo_atual = MODOS['VALORES']
            falar_texto("Modo valores ativado")
        elif key == ord('q') or key == ord('Q'):
            modo_atual = MODOS['QRCODE']
            falar_texto("Modo QR Code ativado")
        elif key == ord('a') or key == ord('A'):
            modo_atual = MODOS['AUTO']
            falar_texto("Modo automático ativado")

except Exception as e:
    logger.error(f"Erro crítico no sistema: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Salvar estatísticas finais
    stats_finais = estatisticas.obter_estatisticas()
    logger.info(f"Estatísticas finais: {stats_finais}")
    
    logger.info("Sistema PayAI finalizado")