import cv2

# classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#video capture setting
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# console message
face_id = input('\n enter user id and press <return> ==> ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = 0
while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 6,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+2,y+h), (255,0,9), 2)
        #inputOutputArray, point1, 2, colorBGR, thickness
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])

    cv2.imshow('image', frame)

    if cv2.waitKey(1) > 0 : break
    elif count >= 1000 : break

print("\n [INFO] Exiting Program and cleanup stuff")

capture.release()
cv2.destroyAllWindows()