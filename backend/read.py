import os

import cv2
from process import preprocess_image, extract_contours, merge_equals, resize_image, make_square_symbol
from predict import predict_out

filename = r'ex6.png' # test image path

# Processing image
inverted_img = preprocess_image(filename)
valid_contours = extract_contours(inverted_img)
valid_contours = merge_equals(valid_contours)
output_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)

# Bound contours to image
for c in valid_contours:
    if c.size == 0:
        continue
    cv2.drawContours(output_img, [c], -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display image
cv2.imshow("Contours", output_img)
cv2.waitKey(0)

# Save images
symbols = []
coordinates = []

for i, c in enumerate(valid_contours):
    if c.size == 0:
        continue
    x, y, w, h = cv2.boundingRect(c)
    coordinates.append(x)
    
    cropped_contour = inverted_img[y:y + h, x:x + w]
    cropped_contour = cv2.bitwise_not(cropped_contour)
    cropped_contour = resize_image(cropped_contour)
    cropped_contour = make_square_symbol(cropped_contour)

    os.makedirs('temp/ex6', exist_ok=True)

    #temp_image_path = f'temp/ex6/temp_contour_{i + 1}.png'
    #cv2.imwrite(temp_image_path, cropped_contour)
    #text = predict_out(temp_image_path, True)
    #print(f"Contour {i + 1}: {text}")
    #output_path = f'temp/ex6/contour_{i + 1}.png'
    #cv2.imwrite(output_path, cropped_contour)

    symbols.append(text)

# Sort (if needed)
#sorted_symbols = [symbol for _, symbol in sorted(zip(coordinates, symbols))]

# Format the LaTeX output
formatted_latex = ""
for i, symbol in enumerate(symbols): #sorted_symbols
    # Check if the current symbol is a LaTeX command
    is_command = symbol.startswith('\\')
    
    # Check if the next symbol is alphabetic and if the current symbol is a command
    needs_space = is_command and i < len(symbols)-1 and symbols[i+1].isalpha()
    
    # Add space after command if needed
    if needs_space:
        formatted_latex += symbol + ' '
    else:
        formatted_latex += symbol
        
print(f"LaTeX Output: {formatted_latex}")