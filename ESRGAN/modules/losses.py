import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19

def get_pixel_loss(criterion='l1'):
    """pixel loss"""
    if criterion == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))

def get_content_loss(criterion='l1', output_layer=54, before_act=True):
    """content loss"""
    if criterion == 'l1':
        loss_func = tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        loss_func = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))
    vgg = VGG19(input_shape=(None, None, 3), include_top=False, weights='imagenet')

    if output_layer == 22:  # Low level feature
        pick_layer = 5
    elif output_layer == 54:  # Hight level feature
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))

    if before_act:
        vgg.layers[pick_layer].activation = None

    fea_extrator = tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)
    fea_extrator.trainable = False

    @tf.function
    def content_loss(real_hr, fake_hr):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        preprocess_fake_hr = preprocess_input(fake_hr * 255.) / 12.75
        preprocess_real_hr = preprocess_input(real_hr * 255.) / 12.75
        fake_hr_features = fea_extrator(preprocess_fake_hr)
        real_hr_features = fea_extrator(preprocess_real_hr)

        return loss_func(real_hr_features, fake_hr_features)

    return content_loss

def get_discriminator_loss(gan_type='ragan'):
    """discriminator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    sigma = tf.sigmoid

    def discriminator_loss_ragan(real_discriminator_logits, fake_discriminator_logits):
        real_logits = sigma(real_discriminator_logits - tf.reduce_mean(fake_discriminator_logits))
        fake_logits = sigma(fake_discriminator_logits - tf.reduce_mean(real_discriminator_logits))
        return 0.5 * (
            cross_entropy(tf.ones_like(real_logits), real_logits) +
            cross_entropy(tf.zeros_like(fake_logits), fake_logits))

    def discriminator_loss(real_discriminator_logits, fake_discriminator_logits):
        real_loss = cross_entropy(tf.ones_like(real_discriminator_logits), sigma(real_discriminator_logits))
        fake_loss = cross_entropy(tf.zeros_like(fake_discriminator_logits), sigma(fake_discriminator_logits))
        return real_loss + fake_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))

def get_generator_loss(gan_type='ragan'):
    """generator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    sigma = tf.sigmoid

    def generator_loss_ragan(real_discriminator_logits, fake_discriminator_logits):
        real_logits = sigma(real_discriminator_logits - tf.reduce_mean(fake_discriminator_logits))
        fake_logits = sigma(fake_discriminator_logits - tf.reduce_mean(real_discriminator_logits))
        return 0.5 * (
            cross_entropy(tf.ones_like(fake_logits), fake_logits) +
            cross_entropy(tf.zeros_like(real_logits), real_logits))

    def generator_loss(real_discriminator_logits, fake_discriminator_logits):
        return cross_entropy(tf.ones_like(fake_discriminator_logits), sigma(fake_discriminator_logits))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
