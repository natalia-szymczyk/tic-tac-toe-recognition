import cv2
import numpy as np
from colorsys import hsv_to_rgb

from colorsys import hsv_to_rgb

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


class Image:
    def __init__(self, _raw):
        self.raw = _raw
        self.contoured = None

    # @property
    # def width(self):
    #     return self.raw.shape[0]
    #
    # @property
    # def height(self):
    #     return self.raw.shape[1]

    @property
    def preprocessed(self):
        kernel = np.ones((4, 4), np.uint8)

        img = cv2.bilateralFilter(self.raw, 25, 75, 75)
        img = channel_selection(img)
        ret, img = cv2.threshold(img, np.mean(img) / 1.2, 255, cv2.THRESH_BINARY)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.bitwise_not(img)
        cv2.imshow('preprocessing', img)

        return img


class Frame(Image):
    def __init__(self, _raw):
        super().__init__(_raw)
        self.boards = []

    def find_boards(self):
        contours, _ = cv2.findContours(self.preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            a = np.min(contour[:, :, 1])
            b = np.max(contour[:, :, 1])
            c = np.min(contour[:, :, 0])
            d = np.max(contour[:, :, 0])

            cv2.imshow("1 Board", self.raw[a:b, c:d])

            cv2.drawContours(self.contoured, [contour], -1, np.array(hsv_to_rgb(1 / len(contours) * i, 1, 1)) * 255.0, 5)
            cv2.imshow("Boards", self.raw)
            board = Board(self.raw[a:b, c:d])
            self.boards.append(board)


class Board(Image):
    def __init__(self, _raw):
        super().__init__(_raw)
        self.tiles = []

    def find_tiles(board, board_tmp):
        contours, hierarchy = cv2.findContours(board_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print(hierarchy)
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


        cv2.imshow("img board", board)


def main(url):
    while True:
        success, raw = cv2.VideoCapture(url).read()

        if not success:
            break

        frame = Frame(raw)
        key   = cv2.waitKey(1)

        cv2.imshow("Tic tac toe", frame.raw)

        if key == 27:
            break
        elif key == 13:
            cv2.imshow("Frame", frame.preprocessed)
            frame.boards

    cv2.destroyAllWindows()


if __name__ == "__main__":
    address = "192.168.1.85"
    port    = "8080"

    main(f"https://{address}:{port}/video")
