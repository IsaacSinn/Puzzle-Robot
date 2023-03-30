import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, TextBox

# import images
solved = cv.imread('./Puzzle Data/Phone Cam/solved_vertical.jpg')
piece = cv.imread('./Puzzle Data/Phone Cam/small.jpg')
assert piece is not None, "file could not be read, check with os.path.exists()"
assert solved is not None, "file could not be read, check with os.path.exists()"
piece = cv.resize(piece, (int(piece.shape[1] * 0.5), int(piece.shape[0] * 0.5)), interpolation=cv.INTER_AREA)

#TODO: make sure piece is 640 x 400


def identify_contour(val, imshow = True):
    # get slider valaue
    canny_low = low_tb.text
    canny_high = high_tb.text

    # greyscale and blur
    piece_grey = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    piece_blur = cv.GaussianBlur(piece_grey, (5,5), 0)

    # Canny edge detection
    edges = cv.Canny(piece_blur, int(canny_low), int(canny_high))

    # clean threshold image
    kernel = np.ones((5,5),np.uint8)
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # show threshold image
    axs[0].imshow(edges, cmap='gray')
    axs[0].set_title("edges")
    axs[1].imshow(closed, cmap='gray')
    axs[1].set_title("closed")

    # find contours
    contours, heirarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # areas = [cv.contourArea(c) for c in contours]
    # max_index = areas.index(max(areas))
    # largest_contour = contours[max_index]

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > 100:
            filtered_contours.append(contour)


    # draw contours
    piece_contour = piece.copy()
    cv.drawContours(piece_contour, filtered_contours, -1, (0,255,0), 3)

    # show contours
    piece_contour = cv.cvtColor(piece_contour, cv.COLOR_BGR2RGB)
    axs[2].set_title("contours")
    axs[2].imshow(piece_contour)

    if imshow:
        fig.canvas.draw_idle()


fig, axs = plt.subplots(1,3, figsize = (20,10))
fig.subplots_adjust(bottom = 0.25)
low_ax = plt.axes([0.2, 0.1, 0.6, 0.05])
high_ax = plt.axes([0.2, 0.05, 0.6, 0.05])
low_tb = TextBox(low_ax, 'canny_low', initial= 50)
high_tb = TextBox(high_ax, 'canny_high', initial = 200)
low_tb.on_submit(identify_contour)
high_tb.on_submit(identify_contour)

plt.show()