import cv2
from colorsys import hsv_to_rgb
import numpy as np
from matplotlib import pyplot as plt
import copy
import imutils


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

    if len(centroids) > 1:
        x1 = centroids[0][0]
        y1 = centroids[0][1]
        x2 = centroids[1][0]
        y2 = centroids[1][1]

        a = (y2 - y1) / (x2 - x1)

        return np.arctan(a) * 180 / 3.14
    else:
        return 0


def find_index(board, centroid):
    contours, hierarchy = cv2.findContours(board, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(contour) for contour in contours]
    contours_areas = sorted(contours_areas, reverse = True)
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

    return 0, 0

    # return sorted_centroids


def find_game_winner(gamestatus):
    if gamestatus[0][0] == gamestatus[0][1] == gamestatus[0][2] and gamestatus[0][0] != "-":
        return f"Zwycięzca: {gamestatus[0][0]}"

    if gamestatus[1][0] == gamestatus[1][1] == gamestatus[1][2] and gamestatus[1][0] != "-":
        return f"Zwycięzca: {gamestatus[1][0]}"

    if gamestatus[2][0] == gamestatus[2][1] == gamestatus[2][2] and gamestatus[2][0] != "-":
        return f"Zwycięzca: {gamestatus[2][0]}"

    if gamestatus[0][0] == gamestatus[1][0] == gamestatus[2][0] and gamestatus[0][0] != "-":
        return f"Zwycięzca: {gamestatus[0][0]}"

    if gamestatus[0][1] == gamestatus[1][1] == gamestatus[2][1] and gamestatus[0][1] != "-":
        return f"Zwycięzca: {gamestatus[0][1]}"

    if gamestatus[0][2] == gamestatus[1][2] == gamestatus[2][2] and gamestatus[0][2] != "-":
        return f"Zwycięzca: {gamestatus[0][2]}"

    if gamestatus[0][0] == gamestatus[1][1] == gamestatus[2][2] and gamestatus[1][1] != "-":
        return f"Zwycięzca: {gamestatus[1][1]}"

    if gamestatus[0][2] == gamestatus[1][1] == gamestatus[2][0] and gamestatus[1][1] != "-":
        return f"Zwycięzca: {gamestatus[1][1]}"

    return "Brak zwycięzcy"


def main(url):
    while True:
        response, img = cv2.VideoCapture(url).read()

        if img is not None:
            cv2.imshow("Tic tac toe", img)

        key = cv2.waitKey(1)

        if key == 27:           # ESC
            break
        elif key == 13:         # Enter
            data = []

            # Image preparation
            img_thresh = preprocessing(img)

            # Find boards on image
            boards = find_boards(img, img_thresh)

            for i, [board, board_thresh] in enumerate(boards):
                # Find approx angle to rotate board
                angle = find_angle(board_thresh)

                # Resize image before rotating
                board_thresh = resize(board_thresh)

                # Rotate image
                board_thresh = imutils.rotate(board_thresh, angle)

                # Find tiles on board
                (board_tiles, gameresult) = find_tiles(board, board_thresh)

                for row in gameresult:
                    print(*row, sep = "|")

                print(find_game_winner(gameresult))

                data.append([board, board_thresh, board_tiles])

            for i, images in enumerate(data):
                for j, image in enumerate(images):
                    plt.subplot(len(data), len(images), (i * len(images)) + j + 1)
                    plt.axis("off")

                    # if ((i * len(images)) + j + 1) % 3 == 0:
                    #     plt.title(find_game_winner(gameresult))

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

    cv2.imshow("test2", img)

    return img


def find_boards(img, img_thresh):
    boards = []
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        if cv2.contourArea(contour) > 10000 and len(approx) == 4:
            cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            boards.append([np.array(img[y_min:y_max, x_min:x_max]), img_thresh[y_min:y_max, x_min:x_max]])

    return boards


def resize(img):
    h, w = img.shape[0:2]
    loss = h // 10
    base_size = h + loss * 2, w + loss * 2, 3
    base = np.zeros(base_size, dtype = np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    base[loss:h + loss, loss:w + loss] = img

    return base


def shape_recognition(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return '-'

    solidity = float(area) / hull_area

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

            tile_x, tile_y = find_index(board_thresh, [cX, cY])

            child = hierarchy[0][i][2]
            children = []

            for item_i, item in enumerate(hierarchy[0]):
                if item[3] == i:
                    children.append(item_i)

            if len(children) > 0:
                children.sort(key = lambda x: cv2.contourArea(contours[x]), reverse = True)
                child = children[0]

            if child > -1 and cv2.contourArea(contours[child]) > 100:
                gamestate[tile_y][tile_x] = shape_recognition(contours[child])

                cv2.drawContours(board_rgb, [contours[child]], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)
                cv2.putText(board_rgb, gamestate[tile_y][tile_x], (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

    return board_rgb, gamestate


if __name__ == "__main__":
    url = 'http://192.168.1.66:8080/video'
    main(url)
