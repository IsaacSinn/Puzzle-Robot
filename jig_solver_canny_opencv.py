import cv2 as cv
import numpy as np

# import images
solved = cv.imread('./Puzzle Data/Phone Cam/solved_vertical.jpg')
piece = cv.imread('./Puzzle Data/Phone Cam/small.jpg')
assert piece is not None, "file could not be read, check with os.path.exists()"
assert solved is not None, "file could not be read, check with os.path.exists()"
piece = cv.resize(piece, (int(piece.shape[1] * 0.5), int(piece.shape[0] * 0.5)), interpolation=cv.INTER_AREA)

#TODO: make sure piece is 640 x 400


def identify_contour(val):
    # get trackbar values
    canny_low = cv.getTrackbarPos('canny_low', 'image')
    canny_high = cv.getTrackbarPos('canny_high', 'image')

    # greyscale and blur
    piece_grey = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    piece_blur = cv.GaussianBlur(piece_grey, (11,11), 0)

    # Canny edge detection
    edges = cv.Canny(piece_blur, canny_low, canny_high)

    # clean threshold image
    kernel = np.ones((5,5),np.uint8)
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # show threshold image
    cv.imshow('blue', piece_blur)
    cv.imshow('closed', closed)

    # find contours
    contours, heirarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # areas = [cv.contourArea(c) for c in contours]
    # max_index = areas.index(max(areas))
    # largest_contour = contours[max_index]

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > 300:
            filtered_contours.append(contour)
    
    print(f"filtered contours: {len(filtered_contours)}")

    # draw contours
    piece_contour = piece.copy()
    cv.drawContours(piece_contour, filtered_contours, -1, (0,255,0), 3)

    # show contours
    cv.imshow('contours', piece_contour)

    cv.waitKey(1)


# create window and trackbars
cv.namedWindow('image')
cv.createTrackbar('canny_low', 'image', 50, 255, identify_contour)
cv.createTrackbar('canny_high', 'image', 200, 255, identify_contour)

# show the initial image
cv.imshow('image', piece)

cv.waitKey(0)
cv.destroyAllWindows()