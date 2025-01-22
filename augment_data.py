import os
import cv2
import numpy as np
import random

base = "data/train"

# Adjusting noise
def add_noise(image):
    # Increase noise level
    level = random.randint(0, 10)
    noise = np.random.normal(0, level, image.shape).astype(np.int16)
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8U)
    return np.clip(noisy_image, 0, 255)


# Adjusting saturation
def adjust_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    factor = random.uniform(0.50, 1.50)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Rotating image
def random_rotate(image):
    (h, w) = image.shape[:2]
    angle = random.randint(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


# Randomly cropping image
def random_crop(image):
    h, w = image.shape[:2]
    crop_h = random.randint(int(0.85 * h), h)
    crop_w = random.randint(int(0.85 * w), w)
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)

    cropped_image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    return cv2.resize(cropped_image, (w, h))  # Resize back to original size


# Apply blur
def apply_gaussian_blur(image):
    ksize = random.choice([3, 5, 7])  # Random kernel sizes for the blur
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


# Applying all transformations
def augment_image(image):
    transforms = [add_noise, adjust_saturation, random_rotate, random_crop, apply_gaussian_blur]
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
