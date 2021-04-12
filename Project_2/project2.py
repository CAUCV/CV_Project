import cv2
import numpy as np

# 이미지 경로
src_img = "flower.PNG"
dst_img = "board.PNG"


global img

global points
points = []

def save_point(x, y):
    point = [x, y]
    points.append(point)

def get_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 6, (0, 255, 0), 1)
        save_point(x, y)

def getHomography():
    M = [[-points[0][0], -points[0][1], -1, 0, 0, 0, points[0][0] * points[4][0], points[0][1] * points[4][0], points[4][0]],
         [0, 0, 0, -points[0][0], -points[0][1], -1, points[0][0] * points[4][1], points[0][1] * points[4][1], points[4][1]],
         [-points[1][0], -points[1][1], -1, 0, 0, 0, points[1][0] * points[5][0], points[1][1] * points[5][0], points[5][0]],
         [0, 0, 0, -points[1][0], -points[1][1], -1, points[1][0] * points[5][1], points[1][1] * points[5][1], points[5][1]],
         [-points[2][0], -points[2][1], -1, 0, 0, 0, points[2][0] * points[6][0], points[2][1] * points[6][0], points[6][0]],
         [0, 0, 0, -points[2][0], -points[2][1], -1, points[2][0] * points[6][1], points[2][1] * points[6][1], points[6][1]],
         [-points[3][0], -points[3][1], -1, 0, 0, 0, points[3][0] * points[7][0], points[3][1] * points[7][0], points[7][0]],
         [0, 0, 0, -points[3][0], -points[3][1], -1, points[3][0] * points[7][1], points[3][1] * points[7][1], points[7][1]]]
    M = np.array(M) # 배열로 변환 (8 x 9)

    # DLT - 교수님이 강조한 부분(1)
    # M = U * D * V.T
    U, D, V_T = np.linalg.svd(M)
    V = V_T.transpose(1,0)
    H = V[:, 8]                 # V의 마지막 열 벡터
    H = np.reshape(H, (3,3))    # 3 x 3으로 reshape

    return H


# img2.shape = (400, 300, 3) = (y, x, z)  매우 헷갈린다... y x z 라니..
# img1 -> img2
def warp(img1, img2, H):
    H_inv = np.linalg.inv(H)
    for y in range(img2.shape[0]): # rows 400
        for x in range(img2.shape[1]): # cols 300
            coor = np.array([x, y, 1]).reshape(3,1) # 좌표를 3x1로 바꿔 (Homogeneous coordinates)
            tmp_coor = np.matmul(H_inv, coor)       # (3x3)*(3x1) --> tmp_coor은 3x1 배열
            tmp_coor = tmp_coor/tmp_coor[2,0]       # 3번째 row 값으로 나눈다 (공식이 그렇다...강의자료에는 없는듯)
            trans_coor = [round(tmp_coor[0,0]), round(tmp_coor[1,0])] #round : 반올림
            tx = round(trans_coor[0])
            ty = round(trans_coor[1])
            a = float(trans_coor[0] - tx)
            b = float(trans_coor[1] - ty)

            # Bilinear interpolation - 교수님이 강조한 부분(2)
            # ty 가 299인 경우가 있어서 강의 자료처럼 +1을 하면 300이 되서 index boundary error가 발생한다.. 왜지?!?
            if tx >= 0 and tx < img2.shape[1] and ty >= 0 and ty < img2.shape[0]:
                for i in range(3):
                    img2[y, x][i] = round((((1.0 - a) * (1.0 - b)) * img1[ty, tx][i])
                                         + ((a * (1.0 - b)) * img1[ty, tx][i])
                                         + ((a * b) * img1[ty, tx][i])
                                         + (((1.0 - a) * b) * img1[ty, tx])[i])



# Read Image1
image_1 = cv2.imread(src_img)
image_1 = cv2.resize(image_1, (300, 400))
cv2.namedWindow('image_1')
cv2.setMouseCallback('image_1', get_point)

# Get corner points to image_1
copy_img1 = image_1.copy()
img = copy_img1
while(1):
    cv2.imshow('image_1',copy_img1)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

# Read Image2
image_2 = cv2.imread(dst_img)
image_2 = cv2.resize(image_2, (300, 400))
cv2.namedWindow('image_2')
cv2.setMouseCallback('image_2', get_point)

# Get corner points to image_2
copy_img2 = image_2.copy()
img = copy_img2
while(1):
    cv2.imshow('image_2',copy_img2)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

print('points: ', points)
homography = getHomography()
warp(image_1, image_2, homography)
cv2.imshow("result", image_2)
cv2.waitKey()
cv2.destroyAllWindows()


