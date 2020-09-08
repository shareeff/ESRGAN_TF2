from pathlib import Path
import tensorflow as tf
from modules.esrgan import rrdb_net


SCALE = 4
INPUT_SHAPE=(None, None, 3)
ALPHA = 0.8

CHECKPOINT_PATH_PSNR = "./saved/checkpoints/psnr"
CHECKPOINT_PATH_ESRGAN = "./saved/checkpoints/esrgan"
SAVE_MODEL_PATH = "./saved/models/interp_esr.h5"
Path(SAVE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

def main():
    
    # define network
    model = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)

    # load checkpoint
    checkpoint_psnr = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(CHECKPOINT_PATH_PSNR):
        checkpoint_psnr.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH_PSNR))
        print("[*] load ckpt psnr from {}.".format(
            tf.train.latest_checkpoint(CHECKPOINT_PATH_PSNR)))
    else:
        print("[*] Cannot find ckpt psnr from {}.".format(
            tf.train.latest_checkpoint(CHECKPOINT_PATH_PSNR)))
        exit()
    vars_psnr = [v.numpy() for v in checkpoint_psnr.model.trainable_variables]

    checkpoint_esrgan = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(CHECKPOINT_PATH_ESRGAN):
        checkpoint_esrgan.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH_ESRGAN))
        print("[*] load ckpt edsr from {}.".format(
            tf.train.latest_checkpoint(CHECKPOINT_PATH_ESRGAN)))
    else:
        print("[*] Cannot find ckpt edsr from {}.".format(
            tf.train.latest_checkpoint(CHECKPOINT_PATH_ESRGAN)))
        exit()
    vars_edsr = [v.numpy() for v in checkpoint_esrgan.model.trainable_variables]

    # network interpolation
    for i, var in enumerate(model.trainable_variables):
        var.assign((1 - ALPHA) * vars_psnr[i] + ALPHA * vars_edsr[i])

    model.save(SAVE_MODEL_PATH)

    


if __name__ == '__main__':
    main()