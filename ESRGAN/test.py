import os
from PIL import Image
from pathlib import Path
import tensorflow as tf 
from tensorflow.keras.models import load_model
from modules.esrgan import rrdb_net
from modules.utils import create_lr_hr_pair, scale_image_0_1_range, tensor2img
from modules.utils import save_image_grid
from modules.metrics import calculate_psnr, calculate_ssim


SCALE = 4
INPUT_SHAPE=(None, None, 3)

FROM_CHECKPOINT = False
#MODEL_PATH = "./saved/models/psnr.h5"
#MODEL_PATH = "./saved/models/esrgan.h5"
MODEL_PATH = "./saved/models/interp_esr.h5"
#CHECKPOINT_PATH = "./saved/checkpoints/psnr"
CHECKPOINT_PATH = "./saved/checkpoints/esrgan"
IMG_PATH = "./images/input_hr/0825.png"
SAVE_IMG_PATH = "./images/results/interp_esr_0825.png"
Path(SAVE_IMG_PATH).parent.mkdir(parents=True, exist_ok=True)

def main():

    model = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)

    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(CHECKPOINT_PATH) and FROM_CHECKPOINT:
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(CHECKPOINT_PATH)))
    else:
        if os.path.isfile(MODEL_PATH):
            h5_model = load_model(MODEL_PATH, custom_objects={'tf': tf})
            weights = h5_model.get_weights()
            model.set_weights(weights)
            print("[*] load model weights from {}.".format(
            MODEL_PATH))
        else:
            print("[*] Cannot find ckpt or h5 model file.")
            exit()

    if os.path.isfile(IMG_PATH):

        lr_image, hr_image = create_lr_hr_pair(IMG_PATH, SCALE) 
        lr_image = scale_image_0_1_range(lr_image)
        lr_image = tf.expand_dims(lr_image, axis=0)
        generated_hr = model(lr_image)
        generated_hr_image = tensor2img(generated_hr)
        unscale_lr_image = tensor2img(lr_image)
        #img = Image.fromarray(generated_hr_image)
        #img.save(SAVE_IMG_PATH)
        psnr = calculate_psnr(hr_image, generated_hr_image)
        ssim = calculate_ssim(hr_image, generated_hr_image)
        print("[***] psnr: {}, ssim: {}".format(psnr, ssim))
        save_image_grid(unscale_lr_image, generated_hr_image, hr_image, save_path=SAVE_IMG_PATH)

    

        














if __name__ == '__main__':
    main()