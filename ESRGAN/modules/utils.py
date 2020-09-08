import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

def read_image(img_path):
    base=os.path.basename(img_path)
    ext = os.path.splitext(base)[1]
    assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
    image = tf.io.read_file(img_path)
    if ext == '.png':
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    return image

def create_lr_hr_pair(img_path, scale):
    image = read_image(img_path)
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


def save_image_grid(lr, hr, ref=None, save_path=None):
    lr_title = "lr: {}".format(lr.shape)
    hr_title = "hr: {}".format(hr.shape)
    images = [lr, hr]
    titles = [lr_title, hr_title]
    if ref is not None:
        ref_title = "ref: {}".format(ref.shape)
        images += [ref]
        titles += [ref_title]
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    else: 
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize = 20)
        axes[i].axis('off')
    fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.25)
    plt.close()