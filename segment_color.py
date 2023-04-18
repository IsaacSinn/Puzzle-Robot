import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math

def identify_contour(img, imshow = False):
    
    # Parameters 
    HSV_LOWER = np.array([6,17,0]) # HSV
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
    contours, heirarchy = cv.findContours(img_seg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # filter contours that are inside other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        if heirarchy[0][i][3] == -1 and cv.contourArea(contour) > MIN_AREA_CONTOUR:
            filtered_contours.append(contour)

    # get largest contour
    largest_contour = sorted(filtered_contours, key=cv.contourArea, reverse=True)[0]

    if imshow:

        # draw contours and minimum bounding rectangle
        img_contour = img.copy()
        cv.drawContours(img_contour, [largest_contour], -1, (0,255,0), 3)
        img_contour = cv.cvtColor(img_contour, cv.COLOR_BGR2RGB) # change to RGB (matplotlib uses RGB)

        # show images
        fig, axs = plt.subplots(1,3, figsize = (20,10))
        axs[0].imshow(img_seg, cmap = 'gray')
        axs[0].set_title("segmented image")
        axs[1].imshow(img_seg_cleaned, cmap = 'gray')
        axs[1].set_title("cleaned segmentation")
        axs[2].imshow(img_contour)
        axs[2].set_title("contours")


        plt.show()
    
    return img, largest_contour

# helper function to order pts tl, tr, br, bl
def __order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    tl, bl = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    br, tr = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")

def minimum_cropped(img, contour, imshow = False):
    
    # find the minimum box rect of the contour
    #TODO: min area quadrilateral instead of rectangle, as it is not always accurate
    rect = cv.minAreaRect(contour)

    box_original = cv.boxPoints(rect)
    box_original = np.int0(box_original)

    # get angle and center of rect
    angle = rect[2]
    if angle > 45:
        angle = angle - 90 # rotate to y axis
    center = rect[0]

    # create a rotation matrix and rotate the image around the center point
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # get box points and rotate box
    pts = np.int0(cv.transform(np.array([box_original]), M))[0]
    pts[pts < 0] = 0
    ordered_pts = __order_points(pts)

    # cropping image
    img_cropped = rotated[ordered_pts[1][1]:ordered_pts[2][1], ordered_pts[0][0]:ordered_pts[1][0]]

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
        fig, axs = plt.subplots(1,3, figsize = (20,10))
        axs[0].imshow(img_box_original)
        #axs[0].set_title("image box contour")
        axs[1].imshow(img_box_rotate)
        #axs[1].set_title("image cropped")
        axs[2].imshow(img_cropped_RGB)
        plt.show()

    return img_cropped

def SIFT_match_solution(img, solution, imshow = False):

    img1 = img.copy()
    img2 = solution.copy()

    sift =  cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    MIN_MATCH_COUNT = 10

    if len(good)>MIN_MATCH_COUNT:
        src_pts_avg = np.float32([ kp1[m.queryIdx].pt for m in good ]).mean(axis=0)
        dst_pts_avg = np.float32([ kp2[m.trainIdx].pt for m in good ]).mean(axis=0)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        d,h,w = img1.shape[::-1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if imshow:
        fig, axs = plt.subplots(1,1, figsize = (20,10))
        img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
        plt.imshow(img3)
        plt.show()

    return [dst_pts_avg]
    
def draw_grid_solution(solution, imshow = False):

    height, width = solution.shape[:2]

    pts = []
    for i in range(width//8, width, width//4):
        for j in range(height//24, height, height//12):
            pts.append([i,j])

    if imshow:
        fig, axs = plt.subplots(1,1, figsize = (20,10))
        solution_draw = solution.copy()
        for pt in pts:
            solution_draw = cv.circle(solution_draw, pt, radius=10, color=(0,0,255), thickness=-1)
        solution_draw = cv.cvtColor(solution_draw, cv.COLOR_BGR2RGB)
        plt.imshow(solution_draw)
        plt.show()
    return pts

def find_puzzle_match(pts, dst_pts, image_hw):

    height, width = image_hw
    
    # find distance helper function
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    closest_distance = float('inf')
    closest_point = None

    for pt in pts:
        dist = distance(pt[0], pt[1], dst_pts[0][0], dst_pts[0][1])
        if dist < closest_distance:
            closest_distance = dist
            closest_point = pt
    
    return [math.ceil(((closest_point[0] - width//8) / (width//4)) + 1), 
            math.ceil(((closest_point[1] - height//24) / (height//12)) + 1)]





def main():

    # import images
    solution = cv.imread('./Puzzle Data/Phone Cam/black mat/solution.jpg')
    piece = cv.imread('./Puzzle Data/Phone Cam/black mat/piece_resize.jpg')
    assert piece is not None, "file could not be read, check with os.path.exists()"
    assert solution is not None, "file could not be read, check with os.path.exists()"

    img_hw = solution.shape[:2] # height, width

    # identify contour and crop puzzle piece
    _, contour = identify_contour(piece, imshow=False)
    cropped_piece = minimum_cropped(piece, contour, imshow=False)

    # identify contour and crop solution puzzle
    _, solution_contour = identify_contour(solution, imshow=False)
    cropped_solution = minimum_cropped(solution, solution_contour, imshow= False)

    pts = draw_grid_solution(cropped_solution, False)

    dst_pts_avg = SIFT_match_solution(cropped_piece, cropped_solution, imshow = True)

    print(find_puzzle_match(pts, dst_pts_avg, img_hw))




if __name__ == "__main__":
    main()