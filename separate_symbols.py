import cv2
import numpy as np

############################# loading + pre-processing image #############################
# loading image
filename = 'D:\Code\latex.ly/test_2.jpg'

# pre-processing image
img = cv2.imread(filename, 0) # grayscale
img = cv2.GaussianBlur(img, (3, 3), 0) # smooth edges using gaussian blur
img = cv2.adaptiveThreshold( # binary image using adaptive threshold
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
img = cv2.bitwise_not(img) # inverting colours to white text on black bg

# display image temporarily
cv2.imshow("pre-processed image", img)
cv2.waitKey(0)

################################## detecting contours ####################################
contours, hierarchy = cv2.findContours( # finding contours of image (of white shapes in black bg)
    img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# reduce number of points, approximating contours to polygons
contours_poly = []
for c in contours:
    contours_poly.append(cv2.approxPolyDP(c, 3, True))
