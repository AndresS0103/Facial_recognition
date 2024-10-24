import cv2
from fer import FER

# Inicializar el detector de emociones
detector = FER()

# Inicializar la webcam
cap = cv2.VideoCapture(0)

while True:
   #Aqui lo que hace es obtener un frame de la camara
    ret, frame = cap.read()

    #Utilizo la funcion para detectar emociones de Fer 
    result = detector.detect_emotions(frame)

    # Aqui se calcula el rectangulo sobre la cara y un texto con la emocion que calcula
    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        emotion_name = max(emotions, key=emotions.get)

        # le paso los parametros a la funcion rectangle para que dibuje el rectangulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Con la funcion putText muestro las emociones que detectó
        cv2.putText(frame, emotion_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow('Detección de Emociones', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# llamar las funciones para que se ejecuten
cap.release()
cv2.destroyAllWindows()

