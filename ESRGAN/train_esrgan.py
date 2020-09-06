from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from modules.esrgan import rrdb_net, discriminator_net
from modules.lr_scheduler import MultiStepLR
from modules.data import load_dataset
from modules.losses import get_pixel_loss, get_content_loss
from modules.losses import get_discriminator_loss, get_generator_loss

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

PIXEL_CRITERION = 'l1'
FEATURE_CRITERION = 'l2'
GAN_TYPE = 'ragan'
WEIGHT_PIXEL = 1e-2
WEIGHT_FEATURE = 1.0
WEIGHT_GAN = 5e-3

HR_HEIGHT = 128
HR_WIDTH = 128
SCALE = 4
BATCH_SIZE = 16
BUFFER_SIZE = 10240
INPUT_SHAPE=(None, None, 3)

NUM_ITER = 400000
SAVE_STEPS =  5000

PRETRAIN_PATH =  "./saved/checkpoints/psnr"
CHECK_POINT_PATH =  "./saved/checkpoints/esrgan"
Path(CHECK_POINT_PATH).mkdir(parents=True, exist_ok=True)
SAVE_GAN_PATH = "./saved/models/esrgan.h5"
Path(SAVE_GAN_PATH).parent.mkdir(parents=True, exist_ok=True)
SAVE_DISC_PATH = "./saved/models/disc_gan.h5"
Path(SAVE_DISC_PATH).parent.mkdir(parents=True, exist_ok=True)


def main():

    dataset = load_dataset(HR_HEIGHT, HR_WIDTH, SCALE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    generator = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)
    discriminator = discriminator_net(input_shape=INPUT_SHAPE)

    learning_rate_G = MultiStepLR(INITIAL_LR_G, LR_STEPS, LR_RATE)
    learning_rate_D = MultiStepLR(INITIAL_LR_D, LR_STEPS, LR_RATE)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate= learning_rate_G,
                                        beta_1= ADAM_BETA1_G,
                                        beta_2= ADAM_BETA2_G
                                        )
    optimizer_D = tf.keras.optimizers.Adam(learning_rate= learning_rate_D,
                                        beta_1= ADAM_BETA1_D,
                                        beta_2= ADAM_BETA2_D
                                        )

    pixel_loss = get_pixel_loss(PIXEL_CRITERION)
    feature_loss = get_content_loss(FEATURE_CRITERION)
    generator_loss = get_generator_loss(GAN_TYPE)
    discriminator_loss = get_discriminator_loss(GAN_TYPE)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     model=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=CHECK_POINT_PATH,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        if tf.train.latest_checkpoint(PRETRAIN_PATH):
            checkpoint.restore(tf.train.latest_checkpoint(PRETRAIN_PATH))
            checkpoint.step.assign(0)
            print("[*] training from pretrain model {}.".format(
                    PRETRAIN_PATH ))
        else:
            print("[*] cannot find pretrain model {}.".format(
                PRETRAIN_PATH))

    @tf.function
    def train_step(lr, hr):
         with tf.GradientTape() as tape:
            generated_hr = generator(lr, training=True)
            real_logits = discriminator(hr, training=True)
            fake_logits = discriminator(generated_hr, training=True)
            losses_G = {}
            losses_D = {}
            losses_G['pixel'] = WEIGHT_PIXEL * pixel_loss(hr, generated_hr)
            losses_G['feature'] = WEIGHT_FEATURE * feature_loss(hr, generated_hr)
            losses_G['gan'] = WEIGHT_GAN * generator_loss(real_logits, fake_logits)
            losses_D['disc'] = discriminator_loss(real_logits, fake_logits)
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

      
        grads_G = tape.gradient(
            total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(
            total_loss_D, discriminator.trainable_variables)
        optimizer_G.apply_gradients(
            zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(
            zip(grads_D, discriminator.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    
    wandb_run_id = "esrgan-training" #@param {type:"string"}
    if HAS_WANDB_ACCOUNT:
        wandb.init(entity='ilab', project=PROJECT, id=wandb_run_id)
    else:
        wandb.init(id=wandb_run_id)
    remain_steps = max(NUM_ITER - checkpoint.step.numpy(), 0)
    pbar = tqdm(total=remain_steps, ncols=50)
    for lr, hr in dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)
        wandb.log({**{"steps": steps},**losses_G, **losses_D, 
            **{"total_loss_G": total_loss_G.numpy()}, 
            **{"learning_rate_G": optimizer_G.lr(steps).numpy(),
            "learning_rate_D": optimizer_D.lr(steps).numpy()}})

        pbar.set_description("loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
            total_loss_G.numpy(), total_loss_D.numpy(),
            optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))
        pbar.update(1)
        if steps % SAVE_STEPS == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))


    generator.save(SAVE_GAN_PATH)
    discriminator.save(SAVE_DISC_PATH)  
             

    









if __name__ == '__main__':
    main()