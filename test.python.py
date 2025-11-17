import cv2

for i in range(5):  # testa os primeiros 5 índices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Câmera encontrada no índice {i}")
        cap.release()
