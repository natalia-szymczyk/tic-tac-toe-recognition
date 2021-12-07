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
    board_rgb = cv2.cvtColor(board, cv2.COLOR_GRAY2RGB)

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
        b = int(y1 - a * x1)

        x_max = np.shape(board_rgb)[1]
        y_max = int(a * x_max + b)

        cv2.circle(board_rgb, (x1, y1), 5, (255, 0, 255), -1)
        cv2.circle(board_rgb, (x2, y2), 5, (255, 0, 255), -1)
        cv2.line(board_rgb, (0, b), (x_max, y_max), (255, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.line(board_rgb, (0, y1), (np.shape(board)[1], y1), (150, 150, 150), thickness=3, lineType=cv2.LINE_AA)

        cv2.imwrite("./chusteczki/chusteczki-3.4.1-prosta.jpg", board_rgb)

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


def find_game_winner(game_status):
    if game_status[0][0] == game_status[0][1] == game_status[0][2] and game_status[0][0] != "-":
        return f"Winner: {game_status[0][0]}"

    if game_status[1][0] == game_status[1][1] == game_status[1][2] and game_status[1][0] != "-":
        return f"Winner: {game_status[1][0]}"

    if game_status[2][0] == game_status[2][1] == game_status[2][2] and game_status[2][0] != "-":
        return f"Winner: {game_status[2][0]}"

    if game_status[0][0] == game_status[1][0] == game_status[2][0] and game_status[0][0] != "-":
        return f"Winner: {game_status[0][0]}"

    if game_status[0][1] == game_status[1][1] == game_status[2][1] and game_status[0][1] != "-":
        return f"Winner: {game_status[0][1]}"

    if game_status[0][2] == game_status[1][2] == game_status[2][2] and game_status[0][2] != "-":
        return f"Winner: {game_status[0][2]}"

    if game_status[0][0] == game_status[1][1] == game_status[2][2] and game_status[1][1] != "-":
        return f"Winner: {game_status[1][1]}"

    if game_status[0][2] == game_status[1][1] == game_status[2][0] and game_status[1][1] != "-":
        return f"Winner: {game_status[1][1]}"

    return "No winner"


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
                # TODO len != 0
                # Find approx angle to rotate board
                angle = find_angle(board_thresh)

                # Resize image before rotating
                board_thresh = resize(board_thresh)

                # Rotate image
                board_thresh = imutils.rotate(board_thresh, angle)
                cv2.imwrite("./chusteczki/chusteczki-3.4.2-obrocony.jpg", board_thresh)

                # Find tiles on board
                (board_tiles, game_result) = find_tiles(board, board_thresh)

                for row in game_result:
                    print(*row, sep = "|")

                print(find_game_winner(game_result))

                data.append([[board, board_thresh, board_tiles, np.zeros((300, 300, 3), dtype = np.uint8)], game_result])

            results = []

            for i, [images, game_result] in enumerate(data):
                bases = make_same_sizes(images)
                bases[3] = draw_game(game_result, bases[3])
                results.append(np.hstack(bases))

            if len(results) > 0:
                final = np.vstack(results)
                cv2.imwrite("./chusteczki/chusteczki-summary.png", final)


    cv2.destroyAllWindows()


def draw_game(game_result, img):
    for i, row in enumerate(game_result):
        text = f"|{row[0]:^3}|{row[1]:^3}|{row[2]:^3}|"
        cv2.putText(img, text, (50, 60 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    winner = find_game_winner(game_result)

    cv2.putText(img, winner, (70, 80 + (i + 1) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


def make_same_sizes(images):
    max_w = np.max([np.shape(image)[0] for image in images])
    max_h = np.max([np.shape(image)[1] for image in images])
    bases = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = np.shape(image)[0:2]
        loss_w = (max_w - w) // 2
        loss_h = (max_h - h) // 2

        base_size = max_w, max_h, 3
        base = np.zeros(base_size, dtype=np.uint8)
        base[loss_w:w + loss_w, loss_h:h + loss_h] = image
        base = cv2.resize(base, (300, 300), interpolation=cv2.INTER_AREA)
        bases.append(base)

    return bases


def preprocessing(img):
    cv2.imwrite("./chusteczki/chusteczki-3.1-oryginalny.jpg", img)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite("./chusteczki/chusteczki-3.2.1-odszumianie.jpg", img)
    img = cv2.bilateralFilter(img, 75, 75, 75)
    cv2.imwrite("./chusteczki/chusteczki-3.2.2-rozmywanie.jpg", img)
    img = channel_selection(img)
    cv2.imwrite("./chusteczki/chusteczki-3.2.3-channel_selection.jpg", img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 3)
    cv2.imwrite("./chusteczki/chusteczki-3.2.4-threshold.jpg", img)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    cv2.imwrite("./chusteczki/chusteczki-3.2.5-dilate.jpg", img)

    return img


def find_boards(img, img_thresh):
    boards = []
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_rgb = copy.deepcopy(img)

    for i, contour in enumerate(contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        cv2.drawContours(img_rgb, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

        child = hierarchy[0][i][2]
        children = []

        for item_i, item in enumerate(hierarchy[0]):
            if item[3] == i and cv2.contourArea(contours[item_i]) > 50:
                children.append(item_i)

        if cv2.contourArea(contour) > 5000 and len(approx) == 4 and len(children) >= 9:
            cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

            x_min = np.min(contour[:, :, 0])
            x_max = np.max(contour[:, :, 0])
            y_min = np.min(contour[:, :, 1])
            y_max = np.max(contour[:, :, 1])

            boards.append([np.array(img[y_min:y_max, x_min:x_max]), img_thresh[y_min:y_max, x_min:x_max]])

    cv2.imwrite("./chusteczki/chusteczki-3.3.1-wszystkie_kontury.jpg ", img_rgb)
    cv2.imwrite("./chusteczki/chusteczki-3.3.2-plansze_kontury.jpg", img)

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

    # board_tiles = copy.deepcopy(board)
    board_rgb = cv2.cvtColor(board_thresh, cv2.COLOR_GRAY2RGB)
    screen_wszystkie = cv2.cvtColor(board_thresh, cv2.COLOR_GRAY2RGB)
    screen_kafelki = cv2.cvtColor(board_thresh, cv2.COLOR_GRAY2RGB)

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

        cv2.drawContours(screen_wszystkie, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)


        if area in tiles and len(approx) == 4:
            cv2.drawContours(screen_kafelki, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

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
                cv2.putText(board_rgb, gamestate[tile_y][tile_x], (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 2)

    cv2.imwrite("./chusteczki/chusteczki-3.5.1-plansza_wszystkie_kontury.jpg", screen_wszystkie)
    cv2.imwrite("./chusteczki/chusteczki-3.5.2-plansza_kafelki.jpg", screen_kafelki)
    cv2.imwrite("./chusteczki/chusteczki-3.6-plansza_ksztalty.jpg", board_rgb)

    return board_rgb, gamestate


if __name__ == "__main__":
    url = 'http://192.168.1.77:8080/video'
    main(url)

