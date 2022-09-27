from mtcnn import MTCNN
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (360, 240))
    detector = MTCNN()
    cv2.imshow("output", img)
    print(detector.detect_faces(img))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
