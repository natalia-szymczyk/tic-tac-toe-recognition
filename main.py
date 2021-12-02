import cv2
from colorsys import hsv_to_rgb
import numpy as np
from matplotlib import pyplot as plt
# import imutils


def index_search(n):
    if n == 1:
        return (0, 0)
    if n == 2:
        return (0, 1)
    if n == 3:
        return (0, 2)
    if n == 4:
        return (1, 0)
    if n == 5:
        return (1, 1)
    if n == 6:
        return (1, 2)
    if n == 7:
        return (2, 0)
    if n == 8:
        return (2, 1)
    if n == 9:
        return (2, 2)


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
    kernel = np.ones((3, 3), np.uint8)

    # img = cv2.bilateralFilter(img, 25, 75, 75)
    img = channel_selection(img)
    # ret, img = cv2.threshold(img, np.mean(img) / 1.2, 255, cv2.THRESH_BINARY)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3)
    img = cv2.medianBlur(img, 9)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.erode(img, kernel, iterations = 1)
    # img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.bitwise_not(img)
    cv2.imshow('preprocessing', img)

    return img

def find_boards(img, img_tmp):
    boards = []
    contours, hierarchy = cv2.findContours(img_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)

    for i, contour in enumerate(contours):
        child = hierarchy[0][i][2]
        # print(hierarchy)

        area = cv2.contourArea(contour)
        # print(i, area)

        if area>10000:
            cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            boards.append([np.array(img[y_min:y_max, x_min:x_max]), img_tmp[y_min:y_max, x_min:x_max]])


    for i, img in enumerate(boards):
        # print(type(img[0]))
        plt.subplot(1, len(boards), i+1)
        img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        result = np.array(img)
        plt.imshow(result)

    plt.show()

    return boards

def shape_recognition(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return '-'

    solidity = float(area) / hull_area

    # fill the gamestate with the right sign
    if (solidity > 0.75):
        return "O"
    else:
        return "X"


def find_tiles(board, board_tmp):
    contours, hierarchy = cv2.findContours(board_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse=True)
    tiles = contours_areas[1:10]

    gamestate = [["-", "-", "-"], ["-", "-", "-"], ["-", "-", "-"]]
    img_width = board_tmp.shape[0]
    img_height = board_tmp.shape[1]
    tileCount = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        cv2.putText(board, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

        if area in tiles:

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            tileCount = tileCount + 1
            child = hierarchy[0][i][2]
            print('rodzic: ', i, ' dzieciak: ', child)

            cv2.drawContours(board, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)\

            # tileX, tileY = index_search(tileCount)
            # tileY = tileCount % 3
            # tileY = (tileCount-1)%3
            # tileX = (tileCount-1)//3
            tileX = round((x / img_width) * 3)
            tileY = round((y / img_height) * 3)

            if child > -1:
                gamestate[tileX][tileY] = shape_recognition(contours[child])

            cv2.putText(board, gamestate[tileX][tileY], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)
            cv2.putText(board, f"{tileX}, {tileY}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, np.array(hsv_to_rgb(i / len(contours), 1, 0.5)) * 255.0, 2)



            plt.subplot(3, 3, tileCount)
            img = cv2.cvtColor(np.array(board[y_min:y_max, x_min:x_max]), cv2.COLOR_BGR2RGB)
            result = np.array(img)
            plt.title(f'{tileCount}: {tileX} {tileY} = {gamestate[tileX][tileY]}')
            plt.axis('off')
            plt.imshow(result)

    plt.show()

    cv2.imshow("img board", board)
    print("Gamestate:")
    for line in gamestate:
        linetxt = ""
        for cel in line:
            linetxt = linetxt + "|" + cel
        print(linetxt)


if __name__ == "__main__":
    url = 'http://192.168.1.77:8080/video'
    main(url)

