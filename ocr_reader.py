import pytesseract

def read_char(image):
    # Configure Tesseract
    # pytesseract.pytesseract.tesseract_cmd = r'D:\Apps\Tesseract-OCR\tesseract.exe'

    # Create custom configuration
    custom_config = f'--oem 1 --psm 10' # 3,6,7,11 psm works for lines, 10 for single characters

    # Load an image
    # image = cv2.imread(image_path)

    text = pytesseract.image_to_string(image, lang = "eng", config=custom_config)
    return text 
