import cv2
from colorsys import hsv_to_rgb
import numpy as np


def channel_selection(img):
    channel_r = img[:, :, 0]
    channel_g = img[:, :, 1]
    channel_b = img[:, :, 2]

    means = [
        np.mean(channel_r),
        np.mean(channel_g),
        np.mean(channel_b)
    ]

    if min(means) == means[0]:
        return channel_r
    elif min(means) == means[1]:
        return channel_g
    else:
        return channel_b

    # cv2.imshow("Processing r", channel_r)
    # cv2.imshow("Processing g", channel_g)
    # cv2.imshow("Processing b", channel_b)


def contrast_correction(img):
    equ = cv2.equalizeHist(img)

    return np.hstack((img, equ))  # stacking images side-by-side


def gamma_correction(img, gamma):
    table = np.empty((1, 256), np.uint8)

    for i in range(256):
        table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv2.LUT(img, table)


def main(url):
    while True:
        response, img = cv2.VideoCapture(url).read()

        if img is not None:
            cv2.imshow("Tic tac toe", img)

        key = cv2.waitKey(1)

        if key == 27:           # ESC
            break
        elif key == 13:         # Enter
            img_tmp = preprocessing(img)
            boards = find_boards(img, img_tmp)

            for [board, board_tmp] in boards:
                find_tiles(board, board_tmp)


    cv2.destroyAllWindows()


def preprocessing(img):
    kernel = np.ones((4, 4), np.uint8)

    img = cv2.bilateralFilter(img, 25, 75, 75)
    img = channel_selection(img)
    ret, img = cv2.threshold(img, np.mean(img) / 1.2, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.bitwise_not(img)
    cv2.imshow('preprocessing', img)

    return img

def find_boards(img, img_tmp):
    boards = []
    contours, hierarchy = cv2.findContours(img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # print(i, area)

        if area > 0:
            cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            boards.append([img[y_min:y_max, x_min:x_max], img_tmp[y_min:y_max, x_min:x_max]])
            # img_board = img[y_min:y_max, x_min:x_max]
            # cv2.imshow('board_tmp', img_board_tmp)
            #
            # find_tiles(img_board)

            # for j, row in enumerate(contour):
            #     for k, value in enumerate(row):
            #         if value[0] == x:
            #             print(i, 'x: ', value, x, j, k)
            #         if value[1] == y:
            #             print(i, 'y: ', value, y, j, k)

    return boards

def find_tiles(board, board_tmp):
    contours, hierarchy = cv2.findContours(board_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
    print(hierarchy)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse=True)
    tiles = contours_areas[1:10]

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area in tiles:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.drawContours(board, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)
            cv2.putText(board, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

        # shape = "unidentified"
        # peri = cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        #
        # if len(approx) == 4:
        #     # compute the bounding box of the contour and use the
        #     # bounding box to compute the aspect ratio
        #     (x, y, w, h) = cv2.boundingRect(approx)
        #     ar = w / float(h)
        #     # a square will have an aspect ratio that is approximately
        #     # equal to one, otherwise, the shape is a rectangle
        #     shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        #     print(shape)


    print(contours_areas)
    print(tiles)
    cv2.imshow("img board", board)


if __name__ == "__main__":
    url = 'http://192.168.1.85 :8080/video'
    main(url)

