import cv2
from colorsys import hsv_to_rgb
import numpy as np


def main(url):
    while True:
        response, img = cv2.VideoCapture(url).read()

        if img is not None:
            cv2.imshow("Tic tac toe", img)

        key = cv2.waitKey(1)

        if key == 27:           # ESC
            break
        elif key == 13:         # Enter
            processing(img)

    cv2.destroyAllWindows()


def processing(img):
    # Initial game state
    state = [
        ["-", "-", "-"],
        ["-", "-", "-"],
        ["-", "-", "-"]
    ]

    # kernel used for noise removal
    kernel = np.ones((7, 7), np.uint8)

    # get the image width and height
    img_width = img.shape[0]
    img_height = img.shape[1]

    # turn into grayscale
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img_b = cv2.medianBlur(img_g, 21)

    # turn into thresholded binary
    ret, thresh1 = cv2.threshold(img_g, 40, 255, cv2.THRESH_BINARY)

    # remove noise from binary
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    img_inv = cv2.bitwise_not(thresh1)


    # find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        cv2.drawContours(img, [contour], -1, np.array(hsv_to_rgb(i / len(contours), 1, 1)) * 255.0, 5)

    cv2.imshow("Processing", img)



if __name__ == "__main__":
    url = 'http://192.168.1.66:8080/video'
    main(url)

