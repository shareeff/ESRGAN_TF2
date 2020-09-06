from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from modules.esrgan import rrdb_net
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

INITIAL_LR = 2e-4
LR_RATE = 0.5
LR_STEPS = [200000, 400000, 600000, 800000]
ADAM_BETA1_G = 0.9
ADAM_BETA2_G = 0.99
W_PIXEL = 1.0
PIXEL_CRITERION = 'l1'

HR_HEIGHT = 128
HR_WIDTH = 128
SCALE = 4
BATCH_SIZE = 16
BUFFER_SIZE = 10240
INPUT_SHAPE=(None, None, 3)

NUM_ITER = 1000000
SAVE_STEPS = 5000


CHECK_POINT_PATH =  "./saved/checkpoints/psnr"
Path(CHECK_POINT_PATH).mkdir(parents=True, exist_ok=True)
SAVE_MODEL_PATH = "./saved/models/psnr.h5"
Path(SAVE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)


def main():

    dataset = load_dataset(HR_HEIGHT, HR_WIDTH, SCALE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    model = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)
    learning_rate = MultiStepLR(INITIAL_LR, LR_STEPS, LR_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate,
                                        beta_1= ADAM_BETA1_G,
                                        beta_2= ADAM_BETA2_G
                                        )
    pixel_loss = get_pixel_loss(PIXEL_CRITERION)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=CHECK_POINT_PATH,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from scratch.")

    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape() as tape:
            generated_hr = model(lr, training=True)
            loss = W_PIXEL * pixel_loss(hr, generated_hr)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    wandb_run_id = "psnr-training" #@param {type:"string"}
    if HAS_WANDB_ACCOUNT:
        wandb.init(entity='ilab', project=PROJECT, id=wandb_run_id)
    else:
        wandb.init(id=wandb_run_id)

    remain_steps = max(NUM_ITER - checkpoint.step.numpy(), 0)
    pbar = tqdm(total=remain_steps, ncols=50)
    for lr, hr in dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        loss = train_step(lr, hr)
        wandb.log({"steps": steps, "loss": loss, "learning_rate": optimizer.lr(steps).numpy()})
        pbar.set_description("loss={:.4f}, lr={:.1e}".format(loss, optimizer.lr(steps).numpy()))
        pbar.update(1)
        if steps % SAVE_STEPS == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))

    model.save(SAVE_MODEL_PATH)


    

    
        
    












if __name__ == '__main__':
    main()