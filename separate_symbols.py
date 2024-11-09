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
inverted_img = cv2.bitwise_not(img) # inverting colours to white text on black bg

# display image temporarily
cv2.imshow("pre-processed image", img)
cv2.waitKey(0)

################################## detecting contours ####################################
contours, hierarchy = cv2.findContours( # finding contours of image (of white shapes in black bg)
    inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# reduce number of points, approximating contours to polygons
contours_poly = []
for c in contours:
    contours_poly.append(cv2.approxPolyDP(c, 3, True))

############################ filtering + validating contours #############################
valid_contours = []
# remove small contours
for c in contours_poly: # c in contours_poly
    if cv2.contourArea(c) > 100:
        valid_contours.append(c)

# sort from left to right
valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

################################### display contours #####################################
output_img = cv2.cvtColor(img, 0) # copy of binary image to display contours

# drawing contours and rectangles around each contour
for c in valid_contours:
    # contour outline
    cv2.drawContours(output_img, [c], -1, (0, 255, 0), 1)

    # bounding rectangle around each contour
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# display contours on original image
cv2.imshow("contours and bounding rectangles", output_img)
cv2.waitKey(0)

############################ extract + save contour as image ##############################
for i, c in enumerate(valid_contours):
    mask = np.zeros(img.shape, dtype=np.uint8) # creating mask with same size as original image
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED) # draw contour on mask
    extracted = cv2.bitwise_and(inverted_img, inverted_img, mask=mask) # extracting contour from image using mask
    x, y, w, h = cv2.boundingRect(c) 
    cropped_contour = extracted[y:y + h, x:x + w] # crop contour using box
    cropped_contour = cv2.bitwise_not(cropped_contour) # invert contours
    cropped_contour = cv2.resize(cropped_contour, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f"D:\Code\latex.ly\ex2 contours\contour_{i+1}.png", cropped_contour) # save cropped contour image

    cv2.imshow(f"contour #{i+1}", cropped_contour) # display contour
    cv2.waitKey(0)


