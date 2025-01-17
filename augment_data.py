import os
import cv2
import numpy as np
import random

base = "data"

def add_noise(image):
    level = random.randint(0, 3)
    noise = np.random.normal(0, level, image.shape).astype(np.int16)
    return cv2.add(image, noise, dtype=cv2.CV_8U)

def adjust_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    factor = random.uniform(0.95, 1.05)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_rotate(image):
    (h, w) = image.shape[:2]
    angle = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    factor = random.randint(0, 20)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_image(image):
    transforms = [add_noise, adjust_saturation, random_rotate, random_brightness]
    random.shuffle(transforms)
    out = image.copy()
    for t in transforms[:random.randint(1, 4)]:
        out = t(out)
    return out

for root, dirs, files in os.walk(base):
    for filename in files:
        if filename.endswith(".png"):
            file_path = os.path.join(root, filename)
            original = cv2.imread(file_path)
            if original is None:
                continue
            name, ext = os.path.splitext(filename)
            for i in range(1, 16):
                aug = augment_image(original)
                new_name = f"{name}_var{i}{ext}"
                new_path = os.path.join(root, new_name)
                cv2.imwrite(new_path, aug)