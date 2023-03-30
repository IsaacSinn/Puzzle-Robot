import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# import images
solved = cv.imread('./Puzzle Data/Phone Cam/solved_vertical.jpg')
piece = cv.imread('./Puzzle Data/Phone Cam/small.jpg')
assert piece is not None, "file could not be read, check with os.path.exists()"
assert solved is not None, "file could not be read, check with os.path.exists()"
# piece = cv.resize(piece, (640, 400), interpolation=cv.INTER_AREA)


def identify_contour(imshow = True):

    # greyscale and blur
    piece_grey = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    piece_blur = cv.GaussianBlur(piece_grey, (5,5),0)
    piece_median = cv.medianBlur(piece_grey, 5)

    # get trackbar position
    # thresh_value = cv.getTrackbarPos('thresh_value', 'controls')

    # OTSU Binarization + binary invert
    _, th1 = cv.threshold(piece_blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # adaptive threshold (mean)
    th2 = cv.adaptiveThreshold(piece_median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 3)

    # clean threshold image
    kernel = np.ones((40,40),np.uint8)
    th1 = cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel)
    kernel_th2 = np.ones((5,5),np.uint8)
    th2 = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel_th2)

    # show threshold image
    fig, axs = plt.subplots(2,3, figsize = (20,10))
    axs[0,0].imshow(piece_blur, cmap='gray')
    axs[0,0].set_title("blur piece")
    axs[0,1].imshow(th1, cmap='gray')
    axs[0,1].set_title("threshold binary image (OTSU)")
    axs[1,1].imshow(th2, cmap = 'gray')
    axs[1,1].set_title("threshold binary image (adaptive mean)")
    

    # detect contours and largest contour
    contours, heirarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

    contours_th2, _ = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour_th2 = sorted(contours_th2, key=cv.contourArea, reverse=True)[0]

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > 1000:
            filtered_contours.append(contour)

    # draw contours
    piece_contour = piece.copy()
    cv.drawContours(piece_contour, [largest_contour], -1, (0,255,0), 3)

    piece_contour_th2 = piece.copy()
    cv.drawContours(piece_contour_th2, [largest_contour_th2], -1, (0,255,0), 3)

    # show contours
    piece_contour = cv.cvtColor(piece_contour, cv.COLOR_BGR2RGB)
    axs[0,2].set_title("contours")
    axs[0,2].imshow(piece_contour)

    piece_contour_th2 = cv.cvtColor(piece_contour_th2, cv.COLOR_BGR2RGB)
    axs[1,2].set_title("contours, adaptive mean")
    axs[1,2].imshow(piece_contour_th2)

    if imshow:
        plt.show()

def main(x):
    identify_contour()

# create trackbars
# cv.namedWindow('controls')

main(1)

cv.waitKey(0)
cv.destroyAllWindows()