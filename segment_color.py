import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



# Parameters 
HSV_LOWER = np.array([6,17,0]) #HSV
HSV_UPPER = np.array([35,115,210])
GAUSSIAN_BLUR = (9,9)
MIN_AREA_CONTOUR = 100


def identify_contour(img, imshow = False):
    img_blur = cv.GaussianBlur(img, GAUSSIAN_BLUR, 0)
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)

    # segmentation using lower and upper bound hsv values
    img_seg = cv.inRange(img_hsv, HSV_LOWER, HSV_UPPER)
    img_seg = 255 - img_seg

    # clean segmentation image
    kernel = np.ones((5,5),np.uint8)
    img_seg_cleaned = cv.morphologyEx(img_seg, cv.MORPH_CLOSE, kernel)

    # detect contours
    contours, heirarchy = cv.findContours(img_seg_cleaned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > MIN_AREA_CONTOUR:
            filtered_contours.append(contour)

    largest_contour = sorted(filtered_contours, key=cv.contourArea, reverse=True)[0]
    
    # draw contours and minimum bounding rectangle
    img_contour = img.copy()
    cv.drawContours(img_contour, [largest_contour], -1, (0,255,0), 3)
    img_contour = cv.cvtColor(img_contour, cv.COLOR_BGR2RGB) # change to RGB (matplotlib uses RGB)

    if imshow:
        # show images
        fig, axs = plt.subplots(1,3, figsize = (10,5))
        axs[0].imshow(img_seg, cmap = 'gray')
        axs[0].set_title("segmented image")
        axs[1].imshow(img_seg_cleaned, cmap = 'gray')
        axs[1].set_title("cleaned segmentation")
        axs[2].imshow(img_contour)
        axs[2].set_title("contours")


        plt.show()
    
    return img, largest_contour

def minimum_cropped(img, contour, imshow = True):
    
    # find the minimum box rect of the contour
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)


    (x, y) = (box[1])
    w = max(box[2][0], box[3][0]) - min(box[0][0], box[1][0])
    h = max(box[0][1], box[3][1]) - min(box[1][1], box[2][1])
    img_cropped = img[y:y+h, x:x+w]

    # draw bounding box
    img_box = img.copy()
    cv.drawContours(img_box, [box], -1, (0,255,0), 3)
    img_box = cv.cvtColor(img_box, cv.COLOR_BGR2RGB)

    if imshow:
        # show images
        fig, axs = plt.subplots(1,3, figsize = (10,5))
        axs[0].imshow(img_box)
        axs[0].set_title("image box contour")
        img_cropped_RGB = cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB)
        axs[1].imshow(img_cropped_RGB)
        axs[1].set_title("image cropped")
        plt.show()

    return img_cropped

def main():

    # import images
    solution = cv.imread('./Puzzle Data/Phone Cam/black mat/solution.jpg')
    piece = cv.imread('./Puzzle Data/Phone Cam/black mat/piece_resize.jpg')
    assert piece is not None, "file could not be read, check with os.path.exists()"
    assert solution is not None, "file could not be read, check with os.path.exists()"

    
    _, contour = identify_contour(piece)
    cropped_piece = minimum_cropped(piece, contour)


if __name__ == "__main__":
    main()



