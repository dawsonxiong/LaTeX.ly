import cv2
import numpy as np

############################# Loading + Pre-processing image #############################
# Loading image
filename = 'D:\Code\latex.ly/test_1.png'

# Pre-processing image
img = cv2.imread(filename, 0)  # Load image in grayscale
if img is None:
    print(f"Error: Could not read image {filename}")
        
img = cv2.GaussianBlur(img, (3, 3), 0)  # Smooth edges using Gaussian blur
img = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10) # Binary image
inverted_img = cv2.bitwise_not(img)  # Invert colors to white text on black background

# Display image temporarily
cv2.imshow("pre-processed image", img)
cv2.waitKey(0)

################################## Detecting contours ####################################
# Finding contours (of white shapes in black bg)
contours, _ = cv2.findContours( 
    inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

############################ Filtering + validating contours #############################
 # Removing small contours
valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

# Sort contours from left to right
valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

################################### Display contours #####################################
output_img = cv2.cvtColor(img, 0) # Copy of binary image to display contours

# Drawing contours and rectangles around each contour
for c in valid_contours:
    # Contour outline
    cv2.drawContours(output_img, [c], -1, (0, 255, 0), 1)

    # Bounding rectangle around each contour
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

################################## Detecting equal signs ####################################
# Note: will add recognition for greater than or equal to / less than or equal to / fractions
def merge_equals(contours):
    y_max = 30
    x_max = 10
    merged_contours = []
    skip = False

    for i in range(len(contours) - 1):
        if skip:
            skip = False
            continue
        
        # Make rectangles for bounds
        r1 = cv2.boundingRect(contours[i])
        r2 = cv2.boundingRect(contours[i + 1])

        # Check if the contours are close enough to be considered an equals sign
        if abs(r1[1] - r2[1]) < y_max and abs(r1[0] - r2[0]) < x_max:
            # Merge rectangles
            x = min(r1[0], r2[0])
            y = min(r1[1], r2[1])
            w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
            h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
            merged_rect = (x, y, w, h)

            # Merging the two contours
            merged_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            merged_contours.append(merged_contour)

            skip = True  # Skip the next contour as it has been merged
        else: # Not equals sign
            merged_contours.append(contours[i])

    if not skip:
        merged_contours.append(contours[-1])

    return merged_contours

valid_contours = merge_equals(valid_contours)

# Display contours on original image
cv2.imshow("contours and bounding rectangles", output_img)
cv2.waitKey(0)

############################ Extract + save contour as image ##############################
for i, c in enumerate(valid_contours):
    mask = np.zeros(img.shape, dtype=np.uint8) # Creating mask with same size as original image
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED) # Draw contour on mask
    extracted = cv2.bitwise_and(inverted_img, inverted_img, mask=mask) # Extracting contour from image using mask
    x, y, w, h = cv2.boundingRect(c) 
    cropped_contour = extracted[y:y + h, x:x + w] # Crop contour using box
    cropped_contour = cv2.bitwise_not(cropped_contour) # Invert contours
    cropped_contour = cv2.resize(cropped_contour, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f"D:\Code\latex.ly\ex1 contours\contour_{i+1}.png", cropped_contour)

    cv2.imshow(f"contour #{i+1}", cropped_contour) # display contour
    cv2.waitKey(0)
