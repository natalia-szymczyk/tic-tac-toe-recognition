import cv2
from colorsys import hsv_to_rgb
import numpy as np
from matplotlib import pyplot as plt
import copy
import imutils
import math


data = []


def channel_selection(img):
    channel_r = img[:, :, 0]
    channel_g = img[:, :, 1]
    channel_b = img[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    means = [
        np.mean(channel_r),
        np.mean(channel_g),
        np.mean(channel_b),
        np.mean(gray)
    ]

    if min(means) == means[0]:
        return channel_r
    elif min(means) == means[1]:
        return channel_g
    elif min(means) == means[2]:
        return channel_b
    else:
        return gray


def find_angle(board):
    contours, hierarchy = cv2.findContours(board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse = True)
    tiles = contours_areas[1:10]
    centroids = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area in tiles:
            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            cX = (x_min + x_max) // 2
            cY = (y_min + y_max) // 2

            centroids.append([cX, cY])

    x1 = centroids[0][0]
    y1 = centroids[0][1]
    x2 = centroids[1][0]
    y2 = centroids[1][1]

    a = (y2 - y1) / (x2 - x1)

    return np.arctan(a) * 180 / 3.14


def find_index(board, centroid):
    contours, hierarchy = cv2.findContours(board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse=True)
    tiles = contours_areas[1:10]
    centroids = []
    sorted_centroids = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area in tiles:
            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            cX = (x_min + x_max) // 2
            cY = (y_min + y_max) // 2

            centroids.append([cX, cY])

    centroids.sort(key = lambda x: x[0])

    for i in range(3):
        tmp = centroids[i*3 : i*3 + 3]
        tmp.sort(key = lambda x: x[1])
        sorted_centroids.append(tmp)

    for i, row in enumerate(sorted_centroids):
        for j, value in enumerate(row):
            if centroid == value:
                return i, j

    # return sorted_centroids


def find_game_winner(gamestatus):
    if gamestatus[0][0] == gamestatus[0][1] == gamestatus[0][2]:
        return f"Zwycięzca: {gamestatus[0][0]}"

    if gamestatus[1][0] == gamestatus[1][1] == gamestatus[1][2]:
        return f"Zwycięzca: {gamestatus[1][0]}"

    if gamestatus[2][0] == gamestatus[2][1] == gamestatus[2][2]:
        return f"Zwycięzca: {gamestatus[2][0]}"

    if gamestatus[0][0] == gamestatus[1][0] == gamestatus[2][0]:
        return f"Zwycięzca: {gamestatus[0][0]}"

    if gamestatus[0][1] == gamestatus[1][1] == gamestatus[2][1]:
        return f"Zwycięzca: {gamestatus[0][1]}"

    if gamestatus[0][2] == gamestatus[1][2] == gamestatus[2][2]:
        return f"Zwycięzca: {gamestatus[0][2]}"

    if gamestatus[0][0] == gamestatus[1][1] == gamestatus[2][2]:
        return f"Zwycięzca: {gamestatus[1][1]}"

    if gamestatus[0][2] == gamestatus[1][1] == gamestatus[2][0]:
        return f"Zwycięzca: {gamestatus[1][1]}"

    return "Remis"


def main(url):
    while True:
        response, img = cv2.VideoCapture(url).read()

        if img is not None:
            cv2.imshow("Tic tac toe", img)

        key = cv2.waitKey(1)

        if key == 27:           # ESC
            break
        elif key == 13:         # Enter
            img_thresh = preprocessing(img)
            boards = find_boards(img, img_thresh)


            for i, [board, board_thresh] in enumerate(boards):
                # Rotate boards
                angle = find_angle(board_thresh)

                board_thresh = resize(board_thresh)
                board_thresh = imutils.rotate(board_thresh, angle)

                # cv2.imshow(f"board {i}", board)
                # cv2.imshow(f"board thresh {i}", board_thresh)

                (board_tiles, gameresult) = find_tiles(board, board_thresh)
                print(find_game_winner(gameresult))
                data.append([board, board_thresh, board_tiles])

            for i, images in enumerate(data):
                for j, image in enumerate(images):
                    # cv2.imshow(f"asd{(i * len(images)) + j + 1}")
                    plt.subplot(len(data), len(images), (i * len(images)) + j + 1)
                    plt.axis("off")
                    if ((i * len(images)) + j + 1) % 3 == 0:
                        plt.title(find_game_winner(gameresult))

                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = np.array(img)

                    plt.imshow(result)

            plt.show()

    cv2.destroyAllWindows()


def preprocessing(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.bilateralFilter(img, 75, 75, 75)
    img = channel_selection(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 3)

    # img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 3)

    # img = cv2.medianBlur(img, 3)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    #
    # background = cv2.morphologyEx(img, cv2.MORPH_DILATE, element)
    # out_gray = cv2.divide(img, background, scale = 255)
    # out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    cv2.imshow("test2", img)

    # img = channel_selection(img)
    # blur = cv2.GaussianBlur(img, (0.5, 0.5), sigmaX = 1, sigmaY = 1)
    # cv2.imshow("test1", blur)
    # divide = cv2.divide(img, blur, scale = 255)
    # cv2.imshow("test2", divide)


    # img = channel_selection(img)
    # img = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    #
    # img = cv2.bilateralFilter(img, 25, 75, 75)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 10, 10)
    # cv2.imshow("test1", img)
    # img = channel_selection(img)
    # cv2.imshow("test2", img)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)
    # img = cv2.medianBlur(img, 3)
    # img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1)

    # kernel = np.ones((7, 7), np.uint8)


    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    #
    # # img = cv2.GaussianBlur(img, (5,5), 0)
    #
    # cv2.imshow("test1", img)
    # img = cv2.bilateralFilter(img, 19, 75, 75)
    # cv2.imshow("test2", img)
    # img = cv2.blur(img, (4, 4))
    # cv2.imshow("test3", img)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    # img = cv2.medianBlur(img, 3)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.bitwise_not(img)
    # cv2.imshow("test4", img)

    # img = cv2.bilateralFilter(img, 25, 75, 75)
    # plt.subplot(2, 3, 1)
    # plt.axis("off")
    # plt.title('blur1')
    # plt.imshow(img, cmap='gray')
    # # cv2.imshow('blur1', img)
    #
    # img = channel_selection(img)
    # plt.subplot(2, 3, 2)
    # plt.axis("off")
    # plt.title('channel selection')
    # plt.imshow(img, cmap='gray')
    # # cv2.imshow('channel selection', img)
    #
    # img = cv2.medianBlur(img, 7)
    # plt.subplot(2, 3, 3)
    # plt.axis("off")
    # plt.title('blur2')
    # plt.imshow(img, cmap='gray')
    # # cv2.imshow('blur2', img)
    #
    # # ret, img = cv2.threshold(img, np.mean(img) / 1.2, 255, cv2.THRESH_BINARY)
    # # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #
    # # ret, img = cv2.threshold(img, np.mean(img) / 1.3, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # plt.subplot(2, 3, 4)
    # plt.axis("off")
    # plt.title('threshold')
    # plt.imshow(img, cmap='gray')
    # # cv2.imshow('threshhold', img)
    #
    # # img = cv2.erode(img, kernel, iterations = 1)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # plt.subplot(2, 3, 5)
    # plt.axis("off")
    # plt.title('morphology')
    # plt.imshow(img, cmap='gray')
    # # cv2.imshow('morph', img)
    #
    # # img = cv2.medianBlur(img, 1)
    # # img = cv2.dilate(img, kernel, iterations = 1)
    # img = cv2.bitwise_not(img)
    # # cv2.imshow('preprocessing', img)
    # plt.subplot(2, 3, 6)
    # plt.axis("off")
    # plt.title('preprocessing')
    # plt.imshow(img, cmap='gray')
    #
    # plt.show()

    return img


def find_boards(img, img_thresh):
    boards = []
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # counters = [0 for item in hierarchy[0]]
    #
    # for i, value in enumerate(hierarchy[0]):
    #     if value[3] != -1:
    #         counters[value[3]] += 1
    #
    # for i, item in enumerate(counters):
    #     if item >= 9:
    #         print(i)

    # contours = contours[counters == 9]

    # list_of_boards = [item[3] != -1 for item in hierarchy[0]]
    # print(list_of_boards)

    for i, contour in enumerate(contours):
        # print(contour)
        child = hierarchy[0][i][2]

        area = cv2.contourArea(contour)

        if area > 10000:
            # cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)
            # cv2.drawContours(img, contours[child], -1, (255, 255, 255), 5)
            # print(area)

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            # cv2.putText(img, str(counters[i]), (x_min, y_max), cv2.FONT_HERSHEY_SIMPLEX, 1.5, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

            boards.append([np.array(img[y_min:y_max, x_min:x_max]), img_thresh[y_min:y_max, x_min:x_max]])
            # cv2.imshow(f'board nr{i}', np.array(img[y_min:y_max, x_min:x_max]))
    # cv2.imshow("asd", img)

    return boards


def resize(img):
    h, w = img.shape[0:2]

    loss = h // 10

    base_size = h + loss * 2, w + loss * 2, 3
    # make a 3 channel image for base which is slightly larger than target img
    base = np.zeros(base_size, dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base[loss:h + loss, loss:w + loss] = img

    return base


def shape_recognition(contour):
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments)

    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return '-'

    solidity = float(area) / hull_area

    print(solidity, *huMoments)

    # fill the gamestate with the right sign
    if solidity > 0.775:
        return "O"
    else:
        return "X"


def find_tiles(board, board_thresh):
    contours, hierarchy = cv2.findContours(board_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse = True)
    tiles = contours_areas[1:10]
    board_tiles = copy.deepcopy(board)
    board_rgb = cv2.cvtColor(board_thresh, cv2.COLOR_GRAY2RGB)

    gamestate = [
        ["-", "-", "-"],
        ["-", "-", "-"],
        ["-", "-", "-"]
    ]

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)

        if area in tiles and len(approx) == 4:
            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            cX = (x_min + x_max) // 2
            cY = (y_min + y_max) // 2

            child = hierarchy[0][i][2]

            # cv2.drawContours(board, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)
            cv2.drawContours(board_rgb, [contours[child]], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)
            cv2.drawContours(board_rgb, [contour], -1, (0, 0, 255), 5)

            tile_x, tile_y = find_index(board_thresh, [cX, cY])

            if child > -1:
                child_area = cv2.contourArea(contours[child])
                # if not(child_area > area * 0.95 or child_area < area * 0.025):
                cv2.drawContours(board_tiles, [contours[child]], -1, (255, 255, 255), 5)
                gamestate[tile_y][tile_x] = shape_recognition(contours[child])
                cv2.putText(board_rgb, gamestate[tile_y][tile_x], (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

    for row in gamestate:
        print(*row)

    return board_rgb, gamestate


if __name__ == "__main__":
    url = 'http://192.168.1.66:8080/video'
    main(url)
