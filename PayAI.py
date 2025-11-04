import cv2
import easyocr
import pyttsx3
import time

# Inicializa leitor OCR (português e inglês)
reader = easyocr.Reader(['pt', 'en'], gpu=False)

# Inicializa o mecanismo de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Inicializa a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Inicializa detector de QRCode
detector = cv2.QRCodeDetector()

ultimo_tempo = time.time()
texto_anterior = ""

print("[INFO] Aponte a câmera para o visor da maquininha ou QR Code...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversão para tons de cinza (melhora leitura)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- LEITURA DE QR CODE ---
    data, bbox, _ = detector.detectAndDecode(gray)
    if data:
        if data != texto_anterior:
            print(f"[QR DETECTADO] {data}")
            engine.say("QR Code detectado")
            engine.runAndWait()
            texto_anterior = data
            ultimo_tempo = time.time()

        if bbox is not None:
            pts = bbox.astype(int).reshape(-1, 2)
            for j in range(len(pts)):
                cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1) % len(pts)]), (0,255,0), 2)
            cv2.putText(frame, "QR Code", (pts[0][0], pts[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # --- LEITURA DO VISOR (OCR) ---
    results = reader.readtext(gray)

    for (bbox, texto, conf) in results:
        if conf > 0.5 and any(c.isdigit() for c in texto):
            agora = time.time()
            if texto != texto_anterior and (agora - ultimo_tempo > 3):
                print(f"[VALOR DETECTADO] {texto} (confiança {conf:.2f})")
                engine.say(f"Valor {texto}")
                engine.runAndWait()
                texto_anterior = texto
                ultimo_tempo = agora

            # Desenha contorno e texto reconhecido na tela
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, texto, (top_left[0], top_left[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Exibe o vídeo com as marcações
    cv2.imshow("Reconhecimento de Maquininha", frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()