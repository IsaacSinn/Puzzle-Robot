import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# import images
solved = cv.imread('./Puzzle Data/Phone Cam/black mat/solution.jpg')
piece = cv.imread('./Puzzle Data/Phone Cam/black mat/piece_resize.jpg')
assert piece is not None, "file could not be read, check with os.path.exists()"
assert solved is not None, "file could not be read, check with os.path.exists()"


# Parameters 
HSV_LOWER = np.array([6,17,0]) #HSV
HSV_UPPER = np.array([35,115,210])
GAUSSIAN_BLUR = (9,9)
MIN_AREA_CONTOUR = 100


def identify_contour(imshow = True):
    piece_blur = cv.GaussianBlur(piece, GAUSSIAN_BLUR, 0)
    piece_hsv = cv.cvtColor(piece_blur, cv.COLOR_BGR2HSV)

    # segmentation using lower and upper bound hsv values
    piece_seg = cv.inRange(piece_hsv, HSV_LOWER, HSV_UPPER)
    piece_seg = 255 - piece_seg

    # clean segmentation image
    kernel = np.ones((5,5),np.uint8)
    piece_seg_cleaned = cv.morphologyEx(piece_seg, cv.MORPH_CLOSE, kernel)

    # detect contours and largest contour
    contours, heirarchy = cv.findContours(piece_seg_cleaned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > MIN_AREA_CONTOUR:
            filtered_contours.append(contour)
    
    # draw contours
    piece_contour = piece.copy()
    cv.drawContours(piece_contour, [largest_contour], -1, (0,255,0), 3)
    piece_contour = cv.cvtColor(piece_contour, cv.COLOR_BGR2RGB) # change to RGB (matplotlib uses RGB)


    # show images
    fig, axs = plt.subplots(1,3, figsize = (20,10))
    axs[0].imshow(piece_seg, cmap = 'gray')
    axs[0].set_title("segmented piece")
    axs[1].imshow(piece_seg_cleaned, cmap = 'gray')
    axs[1].set_title("cleaned segment image")
    axs[2].imshow(piece_contour)
    axs[2].set_title("contours")

    if imshow:
        plt.show()

identify_contour()



