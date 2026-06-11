import time
import re
from payai.config import CONTORNO_TEMPO_VIDA
from payai.logger import logger

# Estado compartilhado
contornos_ativos = []
ultimos_detectados = {}
ultimo_tempo = time.time()
texto_anterior = ""

# ---- conversao numerica PT/ES
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

# ---- validacao e filtragem de valores
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
    p = val.split(',')
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

# ---- repeticao / cooldown
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

# ---- contornos
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

    contornos_ativos.append(((tl, br), texto, agora, tipo))
