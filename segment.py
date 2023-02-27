


import cv2

# Define a callback function for the Canny trackbars
def on_canny_thresh_change(_):
    # Apply Canny edge detection with the current trackbar values
    edges = cv2.Canny(blur, cv2.getTrackbarPos('Canny Min', 'Puzzle Piece'), 
                      cv2.getTrackbarPos('Canny Max', 'Puzzle Piece'))
    cv2.imshow('Puzzle Piece', edges)
    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area (assuming it's the puzzle piece)
    largest_contour = max(contours, key=cv2.contourArea)
    # Draw the contour on the original image in green
    cv2.drawContours(img, [largest_contour], 0, (0, 255, 0), 3)
    # Show the original image with the puzzle piece contour highlighted
    cv2.imshow('Puzzle Piece', img)

# Load the input image
img = cv2.imread('Puzzle Data/frame0033.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Create a window for the trackbars
cv2.namedWindow('Puzzle Piece')

# Add trackbars for adjusting the Canny edge detection parameters
cv2.createTrackbar('Canny Min', 'Puzzle Piece', 100, 500, on_canny_thresh_change)
cv2.createTrackbar('Canny Max', 'Puzzle Piece', 200, 500, on_canny_thresh_change)

# Call the callback function initially to show the puzzle piece with default parameters
on_canny_thresh_change(0)

# Wait for a key press and then exit
cv2.waitKey(0)
cv2.destroyAllWindows()
