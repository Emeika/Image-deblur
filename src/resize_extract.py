import cv2
import os
import numpy as np
import pandas as pd


def resize_images(input_dir, output_dir, size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)


def extract_features(image_dir, label):
    data = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_flatten = img.flatten()
            img_flatten = np.append(img_flatten, label)  # add label at the end
            data.append(img_flatten)
    return data


# directories for sharp and blur images
sharp_dir = 'blur_dataset/sharp'
blur_dir = 'blur_dataset/defocused_blurred'

# directories for resized images
resized_sharp_dir = 'resized_dataset/sharp'
resized_blur_dir = 'resized_dataset/blur'

# resize images to 256x256
resize_images(sharp_dir, resized_sharp_dir, size=(128, 128))
resize_images(blur_dir, resized_blur_dir, size=(128, 128))

# extract features from resized images
sharp_features = extract_features(resized_sharp_dir, label=0)
blur_features = extract_features(resized_blur_dir, label=1)

# combine features
all_features = sharp_features + blur_features

# create a DataFrame and save to CSV
columns = [f'pixel_{i}' for i in range(128*128)] + ['label']
df = pd.DataFrame(all_features, columns=columns)
df.to_csv('image_features.csv', index=False)
