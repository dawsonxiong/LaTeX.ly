import cv2
import numpy as np

# Preprocessing image to be more readable
def preprocess_image(img):
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    inverted_img = cv2.bitwise_not(img)
    denoised = cv2.fastNlMeansDenoising(inverted_img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned


# Extracting contours for each character
def extract_contours(inverted_img):
    padding = 5.7

    contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

    height, width = inverted_img.shape
    adjusted_contours = []

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)

        x_pad = max(x - padding, 0)
        y_pad = max(y - padding, 0)
        w_pad = min(w + 2 * padding, width - x_pad)
        h_pad = min(h + 2 * padding, height - y_pad)

        adjusted_contour = np.array([[x_pad, y_pad], [x_pad + w_pad, y_pad],
                                     [x_pad + w_pad, y_pad + h_pad], [x_pad, y_pad + h_pad]], dtype=np.int32)
        adjusted_contours.append(adjusted_contour)
    return adjusted_contours


# Note: will add support for other similar symbols
# Detecting and dealing with equals sign
def is_equals_sign(w1, h1, w2, h2):
    # Criteria for merging:

    # 1. Widths are similar
    width_similarity = abs(w1 - w2) / max(w1, w2)
    widths_similar = width_similarity < 0.5

    # 2. Heights are small compared to widths (lines)
    h1_ratio = h1 / w1
    h2_ratio = h2 / w2
    heights_small = h1_ratio < 0.5 and h2_ratio < 0.5  # Lines are wider than tall

    return widths_similar and heights_small


def is_dot_or_hat(c1, c2):
    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)
    
    # Calculate centers
    center1_x = x1 + w1/2
    center2_x = x2 + w2/2
    
    # Vertical alignment check
    x_aligned = abs(center1_x - center2_x) < max(w1, w2) * 0.5
    
    # Check if one contour is significantly smaller (dot/hat)
    size_ratio = min(cv2.contourArea(c1), cv2.contourArea(c2)) / max(cv2.contourArea(c1), cv2.contourArea(c2))
    has_small_component = size_ratio < 0.3
    
    # Vertical spacing check
    vertical_gap = abs(y2 - (y1 + h1)) if y1 < y2 else abs(y1 - (y2 + h2))
    reasonable_gap = vertical_gap < max(h1, h2) * 0.5
    
    return x_aligned and has_small_component and reasonable_gap


def merge(contours):
    merged_contours = []
    i = 0
    
    while i < len(contours):
        contour1 = contours[i]
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        
        if i + 1 < len(contours):
            contour2 = contours[i + 1]
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            # Check for equals sign or dotted character
            if is_equals_sign(w1, h1, w2, h2) or is_dot_or_hat(contour1, contour2):
                # Merge the two contours
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                merged_contour = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32)
                
                merged_contours.append(merged_contour)
                i += 2
                continue
        
        merged_contours.append(contour1)
        i += 1
    
    return merged_contours


# Resizing image to a height of 64px
def resize_image(image):
     height = image.shape[0]
     width = image.shape[1]
     dimensions = ((int (width*(64/height))),64)
     return cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)


def make_square_symbol(image):
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)

    symbol_roi = image[y:y + h, x:x + w]
    max_side = max(w, h)
    square_bg = np.ones((max_side, max_side), dtype=np.uint8) * 255

    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2

    square_bg[y_offset:y_offset + h, x_offset:x_offset + w] = symbol_roi

    result = cv2.resize(square_bg, (64, 64), interpolation=cv2.INTER_AREA)
    return result