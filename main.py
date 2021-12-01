import cv2
import numpy

from colorsys import hsv_to_rgb


class Image:
    def __init__(self, _raw):
        self.raw = _raw

    # @property
    # def width(self):
    #     return self.raw.shape[0]
    #
    # @property
    # def height(self):
    #     return self.raw.shape[1]

    @property
    def preprocessed(self):
        img = cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(img, 25, 75, 75)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.bitwise_not(img)

        return img


class Frame(Image):
    @property
    def boards(self):
        contours, _ = cv2.findContours(self.preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j, contour in enumerate(contours):
            a = numpy.min(contour[:, :, 1])
            b = numpy.max(contour[:, :, 1])
            c = numpy.min(contour[:, :, 0])
            d = numpy.max(contour[:, :, 0])
        #
        #     print(a, b, c, d)
        #     img_tmp = self.raw[a:b, c:d]
        #     print(img_tmp)
            cv2.imshow("Display", self.raw[a:b, c:d])

            # cv2.drawContours(self.raw, [contour], -1, numpy.array(hsv_to_rgb(1 / len(contours) * j, 1, 1)) * 255.0, 5)
            # cv2.imshow("Boards", self.raw)

            # potential_boards.append(board)

        # for potential_board in potential_boards:
        #     potential_board.display()
        # return contours


class Board(Image):
    pass


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

    cv2.destroyAllWindows()


if __name__ == "__main__":
    address = "192.168.1.66"
    port    = "8080"

    main(f"https://{address}:{port}/video")
