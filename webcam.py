import cv2
import sys 

# Se carga el clasificador de rostros 
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Puede decidir si realizar la detección sobre un video existente o en tiempo real a través de la cámara web
    # Para este caso, si existe un segundo argumento al ejecutar el script y es un video, se realiza la clasificación en este
    # Si no hay un segundo argumento, se abre la webcam
if len(sys.argv) == 2: #Se decide en base a la cantidad de argumentos 
    videoPath = sys.argv[1]
else:
    videoPath = 0

# Inicia la captura del video desde webcam o analiza un video seleccionado si se ha indicado al ejecutar el script
video_capture = cv2.VideoCapture(videoPath)

print("Presione 'q' para salir")
while True:
    # Inicia un loop infinito que aplica el análisis cuadro por cuadro
    ret, frame = video_capture.read() # .read retorna True o False si es que se está leyendo algún frame 
    #mientras que ret sea True (recibiendo un frame) continúa el análisis cuadro a cuadro)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Se convierte a escala de grises
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=10,
        minSize=(20, 20),
    ) #Mismos factores y argumentos que la función de detección de rostros sobre imágenes
    # Dibujar los rectángulos sobre los rostros
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Muestra el video/captura de webcam con los rectángulos sobre los rostros detectados
    cv2.imshow('Video', frame)
    #cv2.imshow('Frame', gray) muestra el video convertido a escala de grises (si ambos imshow están activados, abre dos ventanas)
    # Para salir del loop presionar "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # Finaliza la función


# Al salir del while anterior, libera el uso de la webcam / cierra el video
video_capture.release()
