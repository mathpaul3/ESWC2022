import os
from mtcnn import MTCNN
import cv2
from djitellopy import Tello
import time

# cpu tensorflow 이용 시 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# tello 및 mtcnn 기본 설정
tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
detector = MTCNN()

# 상수 선언
# face_range = [5400, 6000]
face_range = 7500
pid = [0.5, 0, 0.5]  # pid 상수 [KP, KI, KD]
prev_error = 0

tello.takeoff()
time.sleep(1.5)
tello.send_rc_control(0, 0, 24, 0)
time.sleep(1.0)


def detecting_face(img):
    face_list = []
    face_list_area = []

    faces = detector.detect_faces(img)
    for face in faces:
        x, y, w, h = face["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        center_x = x + w//2
        center_y = y + h//2
        area = w * h
        cv2.circle(img, (center_x, center_y), 4, (0, 255, 0), cv2.FILLED)
        # 사각형 넓이와 사각형 중심 좌표 넣는 리스트
        face_list.append([center_x, center_y])
        face_list_area.append(area)
    # print(face_list_area)
    # print(face_list)
    # 가장 가까운 얼굴 기준으로 텔로가 움직이도록
    if len(face_list_area) != 0:
        # mtcnn detecting 성공했을 경우 최댓값의 index를 추출 (= 빨간색 사각형이 생긴 경우)
        i = face_list_area.index(max(face_list_area))
        return img, [face_list[i], face_list_area[i]]
    else:
        return img, [[0, 0], 0]


def tracking_face(tello, info, pid, p_error, face_range):
    center_x, center_y = info[0]
    window_center_x, window_center_y = 360/2, 200/2
    face_area = info[1]
    fb = 0  # fb 선언

    # yaw 조절
    # error는 중심 x 좌표와 화면의 x 중심 간의 차이
    # error = center_x - 360//2
    fb = int((7500 - face_area) / 300)
    if face_area == 0:
        fb = 0
    ud = int((240/2 - center_y) / 4)
    if center_y == 0:
        ud = 0
    yaw = int((center_x - window_center_x) / 6)
    if center_x == 0:
        yaw = 0
    # print("yaw", yaw, center_x, window_center_x)
    # print("ud", ud, center_y, window_center_y)

    # yaw = int(pid[0] * error + pid[2] * (error - p_error))
    # yaw = int(np.clip(yaw, -90, 90))

    # if center_x == 0:
    #     yaw = 0
    #     error = 0

    # forward/backward 조절
    # if face_area == 0:
    #     fb = 0
    # elif 0 < face_area < face_range:  # 너무 멀다
    #     fb = 10
    # elif face_range < face_area:  # 너무 가깝다
    #     fb = -10

    # print(error, (0, fb, 0, yaw))
    print((0, fb, ud, yaw))
    # left, right / forward, backward / up, down / yaw
    tello.send_rc_control(0, fb, ud, yaw)
    return 0


while True:
    image = tello.get_frame_read().frame
    image = cv2.resize(image, (360, 240))

    image, face_info = detecting_face(image)
    print(face_info)
    # face_info : [[가장 큰 얼굴의 x좌표, 가장 큰 얼굴의 y좌표], 가장 큰 얼굴의 면적] 출력

    prev_error = tracking_face(tello, face_info, pid, prev_error, face_range)

    cv2.imshow('Results', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

tello.streamoff()
cv2.destroyAllWindows()
