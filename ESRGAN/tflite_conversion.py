import tensorflow as tf 
from tensorflow.keras.models import load_model

from modules.esrgan import rrdb_net

SCALE = 4
INPUT_SHAPE=(512, 512, 3)

MODEL_PATH = "./saved/models/interp_esr.h5"
TFLITE_MODEL_PATH = './saved/models/esrgan.tflite'

def main():

    trained_model = load_model(MODEL_PATH, custom_objects={'tf': tf})
    weights = trained_model.get_weights()
    
    model = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)
    model.set_weights(weights)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)













if __name__ == '__main__':
    main()