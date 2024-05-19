import cv2
import os
import numpy as np
import pandas as pd


def resize_images(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)


def extract_features(image_dir):
    data = []
    labels = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img_flatten = img.flatten()
            # get the captcha text from filename
            captcha_text = os.path.splitext(filename)[0]
            # add captcha text at the end
            img_flatten = np.append(img_flatten, captcha_text)
            data.append(img_flatten)
    return data


# directory for captcha images
captcha_dir = 'captcha'

# directory for resized images
resized_captcha_dir = 'resized_captcha_dataset'

# resize images to 128x128
resize_images(captcha_dir, resized_captcha_dir, size=(128, 128))

# extract features from resized images
captcha_features = extract_features(resized_captcha_dir)

# create a DataFrame and save to CSV
columns = [f'pixel_{i}' for i in range(128*128*3)] + ['captcha_text']
df = pd.DataFrame(captcha_features, columns=columns)
df.to_csv('captcha_features.csv', index=False)
