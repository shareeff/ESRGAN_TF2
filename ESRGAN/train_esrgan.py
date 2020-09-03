from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from modules.esrgan import rrdb_net, discriminator_net
from modules.lr_scheduler import MultiStepLR
from modules.data import load_dataset
from modules.losses import get_pixel_loss

HAS_WANDB_ACCOUNT = True
PROJECT = 'esrgan-tf2'
import wandb
if not HAS_WANDB_ACCOUNT:
    wandb.login(anonymous='allow')
else:
    wandb.login()

INITIAL_LR_G = 1e-4
INITIAL_LR_D = 1e-4
LR_RATE = 0.5
LR_STEPS = [50000, 100000, 200000, 300000]
ADAM_BETA1_G = 0.9
ADAM_BETA2_G = 0.99
ADAM_BETA1_D = 0.9
ADAM_BETA2_D = 0.99

HR_HEIGHT = 128
HR_WIDTH = 128
SCALE = 4
BATCH_SIZE = 16
BUFFER_SIZE = 10240
INPUT_SHAPE=(None, None, 3)


def main():

    dataset = load_dataset(HR_HEIGHT, HR_WIDTH, SCALE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    generator = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)
    discriminator = discriminator_net(input_shape=INPUT_SHAPE)

    learning_rate_G = MultiStepLR(INITIAL_LR_G, LR_STEPS, LR_RATE)
    learning_rate_D = MultiStepLR(INITIAL_LR_D, LR_STEPS, LR_RATE)
    optimizer_G = tf.keras.optimizer.Adam(learning_rate= learning_rate_G
                                        beta_1= ADAM_BETA1_G
                                        beta_2= ADAM_BETA2_G
                                        )
    optimizer_D = tf.keras.optimizer.Adam(learning_rate= learning_rate_D
                                        beta_1= ADAM_BETA1_D
                                        beta_2= ADAM_BETA2_D
                                        )

    pixel_loss = get_pixel_loss(PIXEL_CRITERION)

    









if __name__ == '__main__':
    main()