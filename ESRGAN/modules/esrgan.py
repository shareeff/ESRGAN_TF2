import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, PReLU
from tensorflow.keras.layers import BatchNormalization, Concatenate, Lambda, Add


def residual_dense_block(input, filters):
    x1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])

    x5 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, input])

    return x

def rrdb(input, filters):
    x = residual_dense_block(input, filters)
    x = residual_dense_block(x, filters)
    x = residual_dense_block(x, filters)
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])
    return out

def sub_pixel_conv2d(scale_factor=2, **kwargs):
    return Lambda(lambda  x: tf.nn.depth_to_space(x, scale_factor), **kwargs)

def upsample(input_tensor, filters, scale_factor=2):
    x = Conv2D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale_factor=scale_factor)(x)
    x = PReLU(shared_axes=[1,2])(x)
    return x

def rrdb_net(input_shape=(None, None, 3), filters=64, scale_factor=4, name='RRDB_model'):
    lr_image = Input(shape=input_shape, name='input')

    #Pre-residual
    x_start = Conv2D(filters, kernel_size=3, strides=1, padding='same')(lr_image)
    x_start = LeakyReLU(0.2)(x_start)

    #Residual block 
    x = rrdb(x_start, filters)

    #Post Residual block
    x = Conv2D(filters,  kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x, x_start])

    #Upsampling
    x = upsample(x, filters, scale_factor)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    out = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=lr_image, outputs=out, name=name)

def conv2d_block(input, filters, strides=1, bn=True):
    x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
    x = LeakyReLU(0.2)
    if bn:
        x = BatchNormalization(momentum=0.8)(x)
    return x


def discriminator(input_shape=(None, None, 3), filters=64, name='Discriminator'):
    img = Input(shape=input_shape)

    x = conv2d_block(img, filters, bn=False)
    x = conv2d_block(x, filters, strides=2)
    x = conv2d_block(x, filters*2)
    x = conv2d_block(x, filters*2, strides=2)
    x = conv2d_block(x, filters*4)
    x = conv2d_block(x, filters*4, strides=2)
    x = conv2d_block(x, filters*8)
    x = conv2d_block(x, filters*8, strides=2)
    x = Dense(filters*16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=img, outputs=x, name=name)








    
