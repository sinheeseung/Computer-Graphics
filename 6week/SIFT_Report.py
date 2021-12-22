import cv2
import numpy as np
from cv2 import KeyPoint


def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h + ksize, w + ksize))
    pad_img[ksize // 2:h + ksize // 2, ksize // 2:w + ksize // 2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row: row + ksize, col:col + ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst


def my_padding(src, filter):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad] = src

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img


def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    # filter 확인
    # print('<filter>')
    # print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst


def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y


def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy


def SIFT(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("get keypoint")
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    dst[dst < 0.01 * dst.max()] = 0
    dst = find_local_maxima(dst, 21)
    dst = dst / dst.max()

    # harris corner에서 keypoint를 추출
    y, x = np.nonzero(dst)

    keypoints = []
    for i in range(len(x)):
        # x, y, size, angle, response, octave, class_id
        pt_x = int(x[i])  # point x
        pt_y = int(y[i])  # point y
        size = None
        key_angle = -1.
        response = dst[y[i], x[i]]  # keypoint에서 harris corner의 측정값
        octave = 0  # octave는 scale 변화를 확인하기 위한 값 (현재 과제에서는 사용안함)
        class_id = -1
        keypoints.append(KeyPoint(pt_x, pt_y, size, key_angle, response, octave, class_id))

    print('get Ix and Iy...')
    Ix, Iy = calc_derivatives(gray)

    print('calculate angle and magnitude')
    # magnitude / orientation 계산
    magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))
    angle = np.arctan2(Iy, Ix)  # radian 값
    angle = np.rad2deg(angle)  # radian 값을 degree로 변경 > -180 ~ 180도로 표현
    angle = (angle + 180) % 360  # -180 ~ 180을 0 ~ 360의 표현으로 변경

    # keypoint 방향
    print('calculate orientation assignment')
    for i in range(len(keypoints)):
        x, y = keypoints[i].pt
        orient_hist = np.zeros(36, )  # point의 방향을 저장
        for row in range(-8, 8):
            for col in range(-8, 8):
                p_y = int(y + row)
                p_x = int(x + col)
                if p_y < 0 or p_y > src.shape[0] - 1 or p_x < 0 or p_x > src.shape[1] - 1:
                    continue  # 이미지를 벗어나는 부분에 대한 처리
                gaussian_weight = np.exp((-1 / 16) * (row ** 2 + col ** 2))
                orient_hist[int(angle[p_y, p_x] // 10)] += magnitude[p_y, p_x] * gaussian_weight
        ###################################################################
        ## ToDo
        ## orient_hist에서 가중치가 가장 큰 값을 추출하여 keypoint의 angle으로 설정
        ## 가장 큰 가중치의 0.8배보다 큰 가중치의 값도 keypoint의 angle로 설정
        ## keypoint 저장에는 KeyPoint module을 사용할 것
        ## angle은 0 ~ 360도의 표현으로 저장해야 함
        ## np.max, np.argmax를 활용하면 쉽게 구할 수 있음
        ## keypoints[i].angle = ???
        ###################################################################
        max_angle = np.argmax(orient_hist)
        keypoints[i].angle = max_angle * 10.

        max_hist = np.max(orient_hist)
        for j in range(36):
            if orient_hist[j] > max_hist * 0.8:
                keypoints.append(KeyPoint(x, y, size, j * 10, response, octave, class_id))

    print('calculate descriptor')
    descriptors = np.zeros((len(keypoints), 128))  # 8 orientation * 4 * 4 = 128 dimensions

    for i in range(len(keypoints)):
        x, y = keypoints[i].pt
        theta = np.deg2rad(keypoints[i].angle)
        # 키포인트 각도 조정을 위한 cos, sin값
        cos_angle = np.cos(theta)
        sin_angle = np.sin(theta)

        orient_hist8 = np.zeros((16, 8))
        for row in range(-8, 8):
            for col in range(-8, 8):
                # 회전을 고려한 point값을 얻어냄
                row_rot = np.round((cos_angle * col) + (sin_angle * row))
                col_rot = np.round((cos_angle * col) - (sin_angle * row))

                p_y = int(y + row_rot)
                p_x = int(x + col_rot)
                if p_y < 0 or p_y > (src.shape[0] - 1) \
                        or p_x < 0 or p_x > (src.shape[1] - 1):
                    continue
                ###################################################################
                ## ToDo
                ## descriptor을 완성
                ## 4×4의 window에서 8개의 orientation histogram으로 표현
                ## 최종적으로 128개 (8개의 orientation * 4 * 4)의 descriptor를 가짐
                ## gaussian_weight = np.exp((-1 / 16) * (row_rot ** 2 + col_rot ** 2))
                ##################################################################b#
                gaussian_weight = np.exp((-1 / 16) * (row_rot ** 2 + col_rot ** 2))
                descriptor_angle = int((angle[p_y, p_x] - keypoints[i].angle))
                if descriptor_angle < 0:
                    descriptor_angle += 360
                descriptor_angle = descriptor_angle // 45

                if -8 <= col < -4 and -8 <= row < -4:
                    orient_hist8[0, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if -4 <= col < 0 and -8 <= row < -4:
                    orient_hist8[1, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 0 <= col < 4 and -8 <= row < -4:
                    orient_hist8[2, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 4 <= col < 8 and -8 <= row < -4:
                    orient_hist8[3, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight

                if -8 <= col < -4 and -4 <= row < 0:
                    orient_hist8[4, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if -4 <= col < 0 and -4 <= row < 0:
                    orient_hist8[5, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 0 <= col < 4 and -4 <= row < 0:
                    orient_hist8[6, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 4 <= col < 8 and -4 <= row < 0:
                    orient_hist8[7, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight

                if -8 <= col < -4 and 0 <= row < 4:
                    orient_hist8[8, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if -4 <= col < 0 and 0 <= row < 4:
                    orient_hist8[9, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 0 <= col < 4 and 0 <= row < 4:
                    orient_hist8[10, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 4 <= col < 8 and 0 <= row < 4:
                    orient_hist8[11, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight

                if -8 <= col < -4 and 4 <= row < 8:
                    orient_hist8[12, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if -4 <= col < 0 and 4 <= row < 8:
                    orient_hist8[13, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 0 <= col < 4 and 4 <= row < 8:
                    orient_hist8[14, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight
                if 4 <= col < 8 and 4 <= row < 8:
                    orient_hist8[15, descriptor_angle] += magnitude[p_y, p_x] * gaussian_weight

        for j in range(16):
            descriptors[i][j * 8:(j + 1) * 8] = orient_hist8[j]

    return keypoints, descriptors


def main():
    src = cv2.imread("zebra.png")
    src_rotation = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)

    kp1, des1 = SIFT(src)
    kp2, des2 = SIFT(src_rotation)

    ## Matching 부분 ##
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    des1 = des1.astype(np.uint8)
    des2 = des2.astype(np.uint8)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(src, kp1, src_rotation, kp2, matches[:20], outImg=None, flags=2)

    # 결과의 학번 작성하기!
    cv2.imshow('201702033 my_sift', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
