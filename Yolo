import cv2

# Carregar classes
class_names = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Definir classes personalizadas
class Remote:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

class Cup:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

class CellPhone:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

# Inicializar contadores de objetos
remote_counter = Remote()
cup_counter = Cup()
cellphone_counter = CellPhone()

# Carregar vídeo
cap = cv2.VideoCapture(0)

# Definir parâmetros do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Carregar modelo YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Executar detecção de objetos no vídeo
while cap.isOpened():
    _, frame = cap.read()
    classes, scores, boxes = model.detect(frame, confThreshold=0.1, nmsThreshold=0.2)

    for (classId, score, box) in zip(classes, scores, boxes):
        label = class_names[classId[0]]
        if label == 'cup':
            cup_counter.increment()
        elif label == 'remote':
            remote_counter.increment()
        elif label == 'cell phone':
            cellphone_counter.increment()

        cv2.rectangle(frame, box, (255, 0, 0), 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Exibir contagem de objetos
    cv2.putText(frame, f'Remote: {remote_counter.count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f'Cell Phone: {cellphone_counter.count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f'Cup: {cup_counter.count}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
