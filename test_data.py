import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# List of LaTeX symbols to generate images for
latex_symbols = [
    r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9',
    r'+', r'-', r'=', r'\times', r'\pm', r'\leq', r'\geq', r'\neq', 
    r'\approx', r'\propto', r'\in', r'\notin', r'\subset',
    r'\cup', r'\cap', r'\setminus', r'\forall', r'\exists', r'\nexists', r'\emptyset',
    r'\partial', r'\infty', r'\angle', r'\triangle', r'\sim',
    r'\equiv', r'\therefore', r'\Re', r'\Im',
    r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\zeta', r'\theta',
    r'\lambda', r'\mu', r'\nu', r'\xi', r'\pi', r'\rho', r'\sigma',
    r'\tau', r'\upsilon', r'\phi', r'\psi', r'\omega', r'\sum', r'\int', r'\infty'
]

# Directory to save the images
output_dir = r"D:\Code\latex.ly\current_data"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for symbol in latex_symbols:
    plt.figure(figsize=(2, 2))
    plt.text(0.5, 0.5, f"${symbol}$", fontsize=50, ha='center', va='center')
    plt.axis('off')
    filename = symbol.replace("\\", "") + ".png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

# Checking if a contour is an equals sign. can repeat for other similar symbols
def merge_equals(contours):
    y_max = 30
    x_max = 10
    merged_contours = []
    skip = False

    for i in range(len(contours) - 1):
        if skip:
            skip = False
            continue
        
        r1 = cv2.boundingRect(contours[i])
        r2 = cv2.boundingRect(contours[i + 1])

        # Check if the contours are close enough to be considered an equals sign
        if abs(r1[1] - r2[1]) < y_max and abs(r1[0] - r2[0]) < x_max:
            x = min(r1[0], r2[0])
            y = min(r1[1], r2[1])
            w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
            h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
            merged_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            merged_contours.append(merged_contour)
            skip = True  # Skip the next contour as it has been merged
        else:
            merged_contours.append(contours[i])

    if not skip and len(contours) > 0:
        merged_contours.append(contours[-1])

    return merged_contours

# Process each image to apply contours and bounding rectangles
for symbol in latex_symbols:
    filename = symbol.replace("\\", "") + ".png"
    img_path = os.path.join(output_dir, filename)
    img = cv2.imread(img_path, 0)  # Load image in grayscale
    if img is None:
        print(f"Error: Could not read image {img_path}")
        continue
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Smooth edges using Gaussian blur
    img = cv2.adaptiveThreshold(  # Binary image
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    inverted_img = cv2.bitwise_not(img)  # Invert colors to white text on black background

    # Detect contours
    contours, _ = cv2.findContours(
        inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter and validate contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Sort contours from left to right
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    valid_contours = merge_equals(valid_contours)

    # Extract and save contour as image
    for i, c in enumerate(valid_contours):
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        extracted = cv2.bitwise_and(inverted_img, inverted_img, mask=mask)
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = extracted[y:y + h, x:x + w]
        cropped_contour = cv2.bitwise_not(cropped_contour)
        cropped_contour = cv2.resize(cropped_contour, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        filename = symbol.replace("\\", "")
        cv2.imwrite(os.path.join(output_dir, f"{filename}_processed.png"), cropped_contour)

print(f"Processed images for {len(latex_symbols)} LaTeX symbols in '{output_dir}' directory.")
