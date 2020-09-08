import tensorflow as tf

def calculate_psnr(img1, img2):
    #img1 and img2 have range [0, 255]
    #psnr = 20 * np.log10(255.0 / np.sqrt(np.mean((img1 - img2)**2)))
    return tf.image.psnr(img1, img2, max_val=255)

def calculate_ssim(hr, generated_hr):
    #hr and generated_hr have range [0, 255]
    return tf.image.ssim(hr, generated_hr, max_val=255)




