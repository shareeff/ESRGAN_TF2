import os
import numpy as np
import tensorflow as tf 


def create_lr_hr_pair(img_path, scale):
    base=os.path.basename(img_path)
    ext = os.path.splitext(base)[1]
    assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
    image = tf.io.read_file(img_path)
    if ext == '.png':
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    
    lr_height, lr_width = image.shape[0] // scale, image.shape[1] // scale
    hr_height, hr_width = lr_height * scale, lr_width * scale
    hr_image = image[:hr_height, :hr_width, :]
    lr_shape = [lr_height, lr_width]
    lr_image = tf.image.resize(hr_image, lr_shape, method=tf.image.ResizeMethod.BICUBIC)
    
    return lr_image, hr_image

def scale_image_0_1_range(image):
    image = image / 255
    red_max = tf.reduce_max(image, axis=None)
    red_min = tf.reduce_min(image, axis=None)
    if red_max > 1 or red_min < 0:
        image = tf.clip_by_value(
            image, 0, 1, name=None
        )
    return image


def unscale_image_0_255_range(image):
    image = image * 255
    red_max = tf.reduce_max(image, axis=None)
    red_min = tf.reduce_min(image, axis=None)
    if red_max > 255 or red_min < 0:
        image = tf.clip_by_value(
            image, 0, 255, name=None
        )
    return image

def tensor2img(tensor):
    return (np.squeeze(tensor.numpy()).clip(0, 1) * 255).astype(np.uint8)

