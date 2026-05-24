import cv2
from ultralytics import YOLO

# 1. Carrega o modelo YOLO pré-treinado (o 'nano' é super rápido para webcam)
# Na primeira vez que rodar, ele vai baixar o arquivo 'yolov8n.pt' automaticamente
modelo = YOLO("yolov8n.pt")

# 2. Inicializa a webcam
webcam = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

while True:
    sucesso, frame = webcam.read()
    if not sucesso:
        break

    # 3. O YOLO faz toda a mágica aqui: detecta, classifica e desenha na tela
    # 'track' permite que ele siga o objeto se ele se mover
    resultados = modelo.track(frame, persist=True)

    # 4. Renderiza (desenha) as caixas e os nomes dos objetos no frame
    frame_processado = resultados[0].plot()

    # Exibe a imagem na tela
    cv2.imshow("Detector de Objetos Inteligente", frame_processado)

    # Fecha se pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()