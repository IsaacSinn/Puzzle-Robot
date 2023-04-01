import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def identify_contour(img, imshow = False):
    
    # Parameters 
    HSV_LOWER = np.array([6,17,0]) #HSV
    HSV_UPPER = np.array([35,115,210])
    GAUSSIAN_BLUR = (9,9)
    MIN_AREA_CONTOUR = 100

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

    if imshow:

        # draw contours and minimum bounding rectangle
        img_contour = img.copy()
        cv.drawContours(img_contour, [largest_contour], -1, (0,255,0), 3)
        img_contour = cv.cvtColor(img_contour, cv.COLOR_BGR2RGB) # change to RGB (matplotlib uses RGB)

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

def minimum_cropped(img, contour, imshow = False):
    
    # find the minimum box rect of the contour
    rect = cv.minAreaRect(contour)
    box_original = cv.boxPoints(rect)
    box_original = np.int0(box_original)

    # get angle and center of rect
    angle = rect[2]
    if angle > 45:
        angle = 90 - angle
    center = rect[0]

    # create a rotation matrix and rotate the image around the center point
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # get box points and rotate box
    pts = np.int0(cv.transform(np.array([box_original]), M))[0]
    pts[pts < 0] = 0
    print(box_original)
    print(pts)

    # cropping image
    img_cropped = rotated[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    if imshow:
        # draw bounding box for original
        img_box_original = img.copy()
        cv.drawContours(img_box_original, [box_original], -1, (0,255,0), 3)
        img_box_original = cv.cvtColor(img_box_original, cv.COLOR_BGR2RGB)

        # draw bounding box for rotated
        img_box_rotate = rotated.copy()
        cv.drawContours(img_box_rotate, [pts], -1, (0,255,0), 3)
        img_box_rotate = cv.cvtColor(img_box_rotate, cv.COLOR_BGR2RGB)

        # change cropped img to RGB (for matplotlib)
        img_cropped_RGB = cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB)

        # show images
        fig, axs = plt.subplots(1,3, figsize = (10,5))
        axs[0].imshow(img_box_original)
        #axs[0].set_title("image box contour")
        axs[1].imshow(img_box_rotate)
        #axs[1].set_title("image cropped")
        axs[2].imshow(img_cropped_RGB)
        plt.show()

    return img_cropped

def main():

    # import images
    solution = cv.imread('./Puzzle Data/Phone Cam/black mat/solution.jpg')
    piece = cv.imread('./Puzzle Data/Phone Cam/black mat/piece_resize.jpg')
    assert piece is not None, "file could not be read, check with os.path.exists()"
    assert solution is not None, "file could not be read, check with os.path.exists()"

    # identify contour and crop puzzle piece
    # _, contour = identify_contour(piece, imshow=True)
    # cropped_piece = minimum_cropped(piece, contour, imshow=True)

    # identify contour and crop solution puzzle
    _, solution_contour = identify_contour(solution, imshow=True)
    cropped_solution = minimum_cropped(solution, solution_contour, imshow= True)

    # # generate image
    # img = np.zeros((1000, 1000), dtype=np.uint8)
    # img = cv.line(img,(400,400),(511,511),(255,255,255),120)
    # img = cv.line(img,(300,300),(700,500),(255,255,255),120)

    # # find contours / rectangle
    # contours,_= cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # crop = minimum_cropped(img, contours[0], imshow=True)


if __name__ == "__main__":
    main()



