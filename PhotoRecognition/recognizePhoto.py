import cv2
import numpy as np

# Percorso ai file di configurazione e pesi
config_path = 'C:\Informatica\KGP\Python\python_mega\DetectionObject\PhotoRecognition\yoloConf\yolov3.cfg'  # Sostituisci con il tuo percorso
weights_path = 'C:\Informatica\KGP\Python\python_mega\DetectionObject\PhotoRecognition\yoloConf\yolov3.weights'  # Sostituisci con il tuo percorso

print("Inizio del rilevamento degli oggetti...")

# Carica la rete
net = cv2.dnn.readNet(weights_path, config_path)

# Carica l'immagine
img = cv2.imread('C:\Informatica\KGP\Python\python_mega\DetectionObject\PhotoRecognition\images\car.jpg')  # Sostituisci con il tuo percorso

# Preparazione dell'immagine
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Ottieni nomi delle uscite
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Controllo per la compatibilità delle versioni
if len(output_layers_indices.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Fai la previsione
outputs = net.forward(output_layers)

# Elaborazione delle uscite
boxes = []
confidences = []
class_ids = []

# Estrazione delle informazioni di rilevamento
for output in outputs:
    for detection in output:
        scores = detection[5:]  # Ottieni i punteggi per le classi
        class_id = np.argmax(scores)  # Trova l'ID della classe con il punteggio più alto
        confidence = scores[class_id]  # Ottieni la confidenza

        if confidence > 0.5:  # Soglia di confidenza
            center_x = int(detection[0] * img.shape[1])  # Coordinate del centro
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])  # Larghezza e altezza del box
            h = int(detection[3] * img.shape[0])

            # Coordinate del bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])  # Aggiungi il box
            confidences.append(float(confidence))  # Aggiungi la confidenza
            class_ids.append(class_id)  # Aggiungi l'ID della classe

# Applica Non-Maximum Suppression per ridurre il numero di box sovrapposti
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Carica le etichette delle classi (puoi modificare il percorso se necessario)
with open('C:\Informatica\KGP\Python\python_mega\DetectionObject\PhotoRecognition\yoloConf\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Disegna i bounding box e stampa gli oggetti trovati
detected_objects = []  # Lista per tenere traccia degli oggetti rilevati
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        confidence_percentage = confidence * 100  # Convertire in percentuale
        color = (0, 255, 0)  # Verde per il bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Disegna il bounding box
        cv2.putText(img, f'{label} {confidence_percentage:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Disegna l'etichetta
        
        # Aggiungi l'oggetto rilevato alla lista
        detected_objects.append(f"{label} {confidence_percentage:.2f}%")

# Visualizza l'immagine con i bounding box
cv2.imshow("Rilevamento Oggetti", img)
cv2.waitKey(0)  # Aspetta che un tasto venga premuto
cv2.destroyAllWindows()  # Chiudi le finestre

# Stampa gli oggetti trovati
if detected_objects:
    print("Oggetti trovati:")
    for obj in detected_objects:
        print(obj)
else:
    print("Nessun oggetto trovato.")





'''

ESERCIZIO 1
import cv2

# Percorso ai file di configurazione e pesi
config_path = 'C:\Informatica\KGP\Python\python_mega\DailyPythonProject\yolov3.cfg'  # Sostituisci con il tuo percorso
weights_path = 'C:\Informatica\KGP\Python\python_mega\DailyPythonProject\yolov3.weights'  # Sostituisci con il tuo percorso

print("Inizio del rilevamento degli oggetti...")

# Carica la rete
net = cv2.dnn.readNet(weights_path, config_path)

# Carica l'immagine
img = cv2.imread('C:\Informatica\KGP\Python\python_mega\DailyPythonProject\car.jpg')  # Sostituisci con il tuo percorso



# Preparazione dell'immagine
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Ottieni nomi delle uscite
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Controllo per la compatibilità delle versioni
if len(output_layers_indices.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Fai la previsione
outputs = net.forward(output_layers)
print("Oggetti rilevati:", outputs) 

ESERCIZIO 2

import cv2
import numpy as np

# Percorso ai file di configurazione e pesi
config_path = 'C:\\Informatica\\KGP\\Python\\python_mega\\DailyPythonProject\\yolov3.cfg'  # Sostituisci con il tuo percorso
weights_path = 'C:\\Informatica\\KGP\\Python\\python_mega\\DailyPythonProject\\yolov3.weights'  # Sostituisci con il tuo percorso

print("Inizio del rilevamento degli oggetti...")

# Carica la rete
net = cv2.dnn.readNet(weights_path, config_path)

# Carica l'immagine
img = cv2.imread('C:\Informatica\KGP\Python\python_mega\DailyPythonProject\images\cardog.jpg')  # Sostituisci con il tuo percorso

# Preparazione dell'immagine
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Ottieni nomi delle uscite
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Controllo per la compatibilità delle versioni
if len(output_layers_indices.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Fai la previsione
outputs = net.forward(output_layers)

# Elaborazione delle uscite
boxes = []
confidences = []
class_ids = []

# Estrazione delle informazioni di rilevamento
for output in outputs:
    for detection in output:
        scores = detection[5:]  # Ottieni i punteggi per le classi
        class_id = np.argmax(scores)  # Trova l'ID della classe con il punteggio più alto
        confidence = scores[class_id]  # Ottieni la confidenza

        if confidence > 0.5:  # Soglia di confidenza
            center_x = int(detection[0] * img.shape[1])  # Coordinate del centro
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])  # Larghezza e altezza del box
            h = int(detection[3] * img.shape[0])

            # Coordinate del bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])  # Aggiungi il box
            confidences.append(float(confidence))  # Aggiungi la confidenza
            class_ids.append(class_id)  # Aggiungi l'ID della classe

# Applica Non-Maximum Suppression per ridurre il numero di box sovrapposti
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Carica le etichette delle classi (puoi modificare il percorso se necessario)
with open('C:\\Informatica\\KGP\\Python\\python_mega\\DailyPythonProject\\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Disegna i bounding box
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        confidence_percentage = confidence * 100  # Convertire in percentuale
        color = (0, 255, 0)  # Verde per il bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Disegna il bounding box
        cv2.putText(img, f'{label} {confidence_percentage:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Disegna l'etichetta

# Visualizza l'immagine con i bounding box
cv2.imshow("Rilevamento Oggetti", img)
cv2.waitKey(0)  # Aspetta che un tasto venga premuto
cv2.destroyAllWindows()  # Chiudi le finestre


'''