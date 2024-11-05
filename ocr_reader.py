from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\Apps\tesseract\tesseract.exe'

# loading + processing image
image_path = 'D:/Code/test_1.png'
image = cv2.imread(image_path)

# custom tesseract configuration
math_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789+-*/=^()!.x'

# print tesseract OCR's results
text = pytesseract.image_to_string(image, config=math_config, lang='eng')
print(text)
