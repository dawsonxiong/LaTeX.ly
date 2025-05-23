import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from process import preprocess_image, extract_contours, merge, resize_image, make_square_symbol
from predict import predict_out

filename = r'ex6.png' # test image path

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
@app.route('/api/process-latex', methods=['POST'])

# Main function
def process_latex():
    # Get file from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Convert file data to OpenCV image
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    # Processing image
    inverted_img = preprocess_image(img)
    valid_contours = extract_contours(inverted_img)
    valid_contours = merge(valid_contours)

    symbols = []
    coordinates = []

    # Bound contours to image
    for c in valid_contours:
        if c.size == 0:
            continue

        # Draw the contour on the image
        output_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_img, [c], -1, (0, 255, 0), 1)

        # Draw a bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with the contour
        #cv2.imshow("Contours", output_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        x, y, w, h = cv2.boundingRect(c)
        coordinates.append(x)
    
        cropped_contour = inverted_img[y:y + h, x:x + w]
        cropped_contour = cv2.bitwise_not(cropped_contour)
        cropped_contour = resize_image(cropped_contour)
        cropped_contour = make_square_symbol(cropped_contour)

        text = predict_out(cropped_contour, is_path=False)

        #os.makedirs('temp/ex6', exist_ok=True)
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
    if not symbols:
        print("No symbols detected.")
    else:
        for i, symbol in enumerate(symbols): #sorted_symbols
            # Current symbol is a LaTeX command
            is_command = symbol.startswith('\\')
            
            # Next symbol is alphabetic and if the current symbol is a command
            needs_space = is_command and i < len(symbols)-1 and symbols[i+1].isalpha()
            
            if needs_space:
                formatted_latex += symbol + ' '
            else:
                formatted_latex += symbol + ' '
        
    print(f"LaTeX Output: {formatted_latex}")

    # Return JSON response
    return jsonify({
        'latex_output': formatted_latex,
        'symbols': symbols
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)