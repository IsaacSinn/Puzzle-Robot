import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# import images
solved = cv.imread('./Puzzle Data/Phone Cam/solved_vertical.jpg')
piece = cv.imread('./Puzzle Data/Phone Cam/small.jpg')
assert piece is not None, "file could not be read, check with os.path.exists()"
assert solved is not None, "file could not be read, check with os.path.exists()"
piece = cv.resize(piece, (int(piece.shape[1] * 0.5), int(piece.shape[0] * 0.5)), interpolation=cv.INTER_AREA)

#TODO: make sure piece is 640 x 400


def identify_contour(imshow = True):

    # greyscale and blur
    piece_grey = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    piece_blur = cv.GaussianBlur(piece_grey, (5,5), 0)

    # get trackbar position
    # thresh_value = cv.getTrackbarPos('thresh_value', 'controls')

    # OTSU Binarization
    _, th1 = cv.threshold(piece_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Adaptive threshold

    # clean threshold image
    kernel = np.ones((60,60),np.uint8)
    th1 = cv.erode(th1, kernel, iterations=1)
    th1 = cv.dilate(th1, kernel, iterations=1)

    # show threshold image
    fig, axs = plt.subplots(1,3, figsize = (20,10))
    axs[0].imshow(piece_blur, cmap='gray')
    axs[0].set_title("blur piece")
    axs[1].imshow(th1, cmap='gray')
    axs[1].set_title("threshold")

    # find contours
    contours, _ = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    # return largest_contour

    # draw contours
    piece_contour = piece.copy()
    cv.drawContours(piece_contour, contours, -1, (0,255,0), 3)

    # show contours
    piece_contour = cv.cvtColor(piece_contour, cv.COLOR_BGR2RGB)
    axs[2].set_title("contours")
    axs[2].imshow(piece_contour)


    if imshow:
        plt.show()

def main(x):
    identify_contour()

# create trackbars
# cv.namedWindow('controls')

main(1)

cv.waitKey(0)
cv.destroyAllWindows()