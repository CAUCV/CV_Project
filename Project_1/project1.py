import cv2
import numpy as np

global img

global num # image에 숫자를 찍기 위한 전역변수
num = 1

# 마우스 이벤트를 통해 추출된 x,y 좌표값을 저장할 리스트
global points
points = []

def save_point(x, y):
    point = [x, y]
    points.append(point)

def get_point(event, x, y, flags, param):
    global num
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 6, (0, 0, 255), 3)
        save_point(x, y)
        cv2.putText(img, str(num), (x-30, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
        num = num + 1


image_1 = cv2.imread("1st.jpg")
image_1 = cv2.resize(image_1, (700, 700))
cv2.namedWindow('image_1')
cv2.setMouseCallback('image_1', get_point)

# image_1
img = image_1
while(1):
    cv2.imshow('image_1',image_1)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


image_2 = cv2.imread("2nd.jpg")
image_2 = cv2.resize(image_2, (700, 700))
cv2.namedWindow('image_2')
cv2.setMouseCallback('image_2', get_point)

# image_2
img = image_2
while(1):
    cv2.imshow('image_2',image_2)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

# 0~3:image1의 keypoints / 4~7:image2의 keypoints
print('points: ', points)
img1_points = points[0:4]
img2_points = points[4:8]