import cv2

# 1. Carrega o arquivo XML com o modelo pré-treinado de detecção facial
# O OpenCV já vem com esse arquivo embutido
classificador_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Inicializa a webcam (0 geralmente é a webcam integrada do notebook)
webcam = cv2.VideoCapture(0)

print("Pressione 'q' para sair do programa.")

while True:
    # Captura o frame atual da webcam
    sucesso, frame = webcam.read()
    
    if not sucesso:
        print("Não foi possível acessar a webcam.")
        break

    # O OpenCV trabalha melhor com imagens em escala de cinza para detecção
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detecta os rostos na imagem
    # scaleFactor: compensa rostos mais próximos ou distantes da câmera
    # minNeighbors: quantos vizinhos cada retângulo candidato deve ter para ser mantido
    rostos = classificador_face.detectMultiScale(frame_cinza, scaleFactor=1.3, minNeighbors=5)

    # 4. Desenha um retângulo ao redor de cada rosto detectado
    for (x, y, largura, altura) in rostos:
        # cv2.rectangle(imagem, (ponto_inicial), (ponto_final), (cor_BGR), espessura)
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)

    # Exibe o frame com o retângulo na tela
    cv2.imshow('Detector de Rostos na Webcam', frame)

    # Se o usuário pressionar a tecla 'q', o loop fecha
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Libera a webcam e fecha as janelas abertas
webcam.release()
cv2.destroyAllWindows()