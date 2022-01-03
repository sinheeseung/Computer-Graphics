import numpy as np
import cv2
import random

def my_bilinear(img, x, y):
    floorX, floorY = int(x), int(y)
    (h, w,c) = img.shape
    t, s = x - floorX, y - floorY

    zz = (1 - t) * (1 - s)
    zo = t * (1 - s)
    oz = (1 - t) * s
    oo = t * s

    if floorX >= h-2:
        floorX = h-2
    if floorY >= w-2:
        floorY = w-2

    interVal = img[floorY, floorX, :] * zz + img[floorY, floorX + 1, :] * zo + \
               img[floorY + 1, floorX, :] * oz + img[floorY + 1, floorX + 1, :] * oo

    return interVal
def my_gaussian(size, sigma):
    h,w = size
    y, x = np.mgrid[-(h // 2):(w // 2) + 1, -(h // 2):(w // 2) + 1]
    # 2차 gaussian mask 생성
    filter_gaus = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    filter_gaus /= np.sum(filter_gaus)
    return filter_gaus

def backward(img1, M):
    h, w, c = img1.shape
    result = np.zeros((h*2, w*2, c))

    filter_gaus = my_gaussian((5,5), 1)
    for row in range(h*2):
        for col in range(w*2):
            xy_prime = np.array([[col, row, 1]]).T
            xy = (np.linalg.inv(M)).dot(xy_prime)

            x_ = xy[0, 0]
            y_ = xy[1, 0]

            if x_ < 0 or y_ < 0 or (x_ + 1) >= w or (y_ + 1) >= h:
                continue

            for mask_row in range(-2, 3):
                for mask_col in range(-2, 3):
                    result[row, col, :] += my_bilinear(img1, x_+mask_col, y_+mask_row) * filter_gaus[mask_col + 2, mask_row + 2]

    return result
def my_ls(matches, kp1, kp2):
    A = []
    B = []
    for idx, match in enumerate(matches):
        trainInd = match.trainIdx
        queryInd = match.queryIdx
        x,y = kp1[queryInd].pt
        x_prime,y_prime = kp2[trainInd].pt

        A.append([x,y,1,0,0,0])
        A.append([0,0,0,x,y,1])
        B.append([x_prime])
        B.append([y_prime])
    A = np.array(A)
    B = np.array(B)

    try:
        ATA = np.dot(A.T, A)
        ATB = np.dot(A.T, B)
        X = np.dot(np.linalg.inv(ATA), ATB)
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X

def get_matching_keypoints(img1, img2, keypoint_num):
    sift = cv2.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.DIST_L2)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches
def feature_matching_RANSAC(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=10):
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    matches_shuffle = matches.copy()

    inliers = [] #랜덤하게 고른 n개의 point로 구한 inlier개수 결과를 저장
    M_list = [] #랜덤하게 고른 n개의 point로 만든 affine matrix를 저장
    for i in range(iter_num):
        print('\rcalculate RANSAC ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        #######################################################################
        # ToDo
        # RANSAC을 이용하여 최적의 affine matrix를 찾고 변환하기
        # 1. 랜덤하게 3개의 matches point를 뽑아냄
        # 2. 1에서 뽑은 matches를 이용하여 affine matrix M을 구함
        # 3. 2에서 구한 M을 모든 matches point와 연산하여 inlier의 개수를 파악
        # 4. iter_num 반복하여 가장 많은 inlier를 가지는 M을 최종 affine matrix로 채택
        ########################################################################
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]
        X = my_ls(three_points, kp1, kp2)
        M = np.array([[X[0][0], X[1][0], X[2][0]],
                      [X[3][0], X[4][0], X[5][0]],
                      [0, 0, 1]])
        count = 0
        M_list.append(M)
        for idx, match in enumerate(matches):
            #모든 match_point 반복
            trainInd = match.trainIdx
            queryInd = match.queryIdx

            kp1_x, kp1_y = kp1[queryInd].pt
            kp2_x, kp2_y = kp2[trainInd].pt

            xy = np.array([kp1_x, kp1_y, 1]).T
            xy_prime = np.dot(M,xy)

            if L2_distance(xy_prime[:2], (kp2_x,kp2_y)) < threshold_distance:
                # 모든 match_point에 대조하여 두 점 사이 거리가 10 이하인 경우
                count += 1

        inliers.append(count)
    inliers = np.array(inliers)
    best_M = M_list[inliers.argmax()]
    result = backward(img1, best_M)

    return result.astype(np.uint8)

def L2_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1-vector2)**2))

def main():
    src = cv2.imread('Lena.png')
    src = cv2.resize(src, None, fx=0.5, fy=0.5)
    src2 = cv2.imread('LenaFaceShear.png')
    result_RANSAC = feature_matching_RANSAC(src, src2)
    cv2.imshow('input', src)
    cv2.imshow('gaussian backward 201702033', result_RANSAC)
    cv2.imshow('goal', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()