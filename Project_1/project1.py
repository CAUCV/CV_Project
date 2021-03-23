import cv2
import matplotlib.pyplot as plt
import numpy as np

global img
global num # image에 숫자를 찍기 위한 전역변수
num = 1

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

def get_patches(size):
    img_patches = []
    for i in range(4):
        img_patches.append(image_1[points[i][1] - size: points[i][1] + size, points[i][0] - size: points[i][0] + size])
    for i in range(4, 8):
        img_patches.append(image_2[points[i][1] - size: points[i][1] + size, points[i][0] - size: points[i][0] + size])

    return img_patches


def get_histogram(img_patches, bin):
    patch_hist = []
    for i in range(8):
        dx = cv2.Sobel(img_patches[i], cv2.CV_32F, 1, 0, ksize=1)
        dy = cv2.Sobel(img_patches[i], cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True) # 벡터의 크기, 각도
        #mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)

        hist = cv2.calcHist([angle], [0], None, [bin], [0, 360])
        hist = hist.flatten()
        # plt.hist(angle.reshape(-1), weights=mag.reshape(-1), bins=10, range=(0,360))
        patch_hist.append(hist)

    return patch_hist


def draw_histogram(patch_hist, bin):
    fig, ax = plt.subplots(2, 4, figsize=(20,15))
    bin_x = np.arange(bin)

    for i in range(4):
        ax[0, i].bar(bin_x, patch_hist[i], width=1, color='k')
        ax[0, i].set_title("[image_1] NO." + str(i+1))
    for i in range(4):
        ax[1, i].bar(bin_x, patch_hist[i+4], width=1, color='b')
        ax[1, i].set_title("[image_2] NO." + str(i+5))
    plt.show()

def compare_histogram(patch_hist):
    result = []
    for a in range(4):
        temp_dist = []
        for b in range(4,8):
            #temp_dist.append(cv2.compareHist(patch_hist[a], patch_hist[b], cv2.HISTCMP_BHATTACHARYYA))
            #print(cv2.compareHist(patch_hist[a], patch_hist[b], cv2.HISTCMP_BHATTACHARYYA))

            #temp_dist.append(cv2.compareHist(patch_hist[a], patch_hist[b], cv2.HISTCMP_CHISQR))
            #print(cv2.compareHist(patch_hist[a], patch_hist[b], cv2.HISTCMP_CHISQR))

            temp_dist.append(np.linalg.norm(patch_hist[a]-patch_hist[b]))
            print(np.linalg.norm(patch_hist[a]-patch_hist[b]))
        result.append(temp_dist.index(min(temp_dist)))
    return result

# Read Image1
image_1 = cv2.imread("1st.jpg")
image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
ratio = 600.0 / image_1.shape[1]
dim = (600, int(image_1.shape[0] * ratio))
image_1 = cv2.resize(image_1, dim)

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
image_2 = cv2.imread("2nd.jpg")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
ratio = 600.0 / image_2.shape[1]
dim = (600, int(image_2.shape[0] * ratio))
image_2 = cv2.resize(image_2, dim)
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

# Get corner patches
img_patches = get_patches(size=25)

# Get histograms
bin = 36
patch_hist = get_histogram(img_patches, bin)
draw_histogram(patch_hist, bin)
result = compare_histogram(patch_hist)
print(result)
for i in range(4):
    print(i+1,"번,",result[i]+5,"번")
