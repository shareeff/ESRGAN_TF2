import os
import random
import tensorflow as tf
import numpy as np


DATA_PATH = "/media/shareef/MLDev/Datasets/DIV2K/DIV2K_train_HR"

def scale_input_image(img):
    #img/ 255.
    return tf.image.convert_image_dtype(img, dtype=tf.float32)

def unscale_output_image(img):
    #img * 255
    return tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)

def random_crop_and_flip(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    image = img[y:(y+dy), x:(x+dx), :]
    flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
    if(tf.equal(flip_case, 0)):
        image = tf.image.flip_left_right(image)
    return image

def load_and_preprocess_image(image_path, hr_height, hr_width, crop_per_image, ext):
    assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
    image = tf.io.read_file(image_path)
    if ext == '.png':
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)

    image = scale_input_image(image)
    cropped_images = [ random_crop_and_flip(image, (hr_height, hr_width)) for _ in range(crop_per_image)] 

    return cropped_images



def load_dataset(hr_height, hr_width, scale, crop_per_image=20, ext='.png'):
    image_paths = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if f'{ext}' in file:
                image_paths.append(os.path.join(root, file))
    
    random.shuffle(image_paths)
    images = []
    for img_path in image_paths:
        images += load_and_preprocess_image(img_path, hr_height, hr_width, crop_per_image, ext)

    random.shuffle(images)
    hr_images = []
    lr_images = []
    for img in images:
        hr_image = img
        lr_shape = [int(hr_image.shape[0]/scale), int(hr_image.shape[1]/scale)]
        lr_image = tf.image.resize(hr_image, lr_shape, method=tf.image.ResizeMethod.BICUBIC)
        #lr_image = lr_image / 255
        lr_image = tf.clip_by_value(
        lr_image, 0, 1, name=None
        )
        hr_images.append(hr_image)
        lr_images.append(lr_image)

    lr_dataset = tf.data.Dataset.from_tensor_slices(lr_images)
    hr_dataset = tf.data.Dataset.from_tensor_slices(hr_images)

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    return dataset

    


