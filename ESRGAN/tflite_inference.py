from PIL import Image
import numpy as np
import tensorflow as tf
from modules.utils import read_image, scale_image_0_1_range

INPUT_SHAPE= [512, 512]

TFLITE_MODEL_PATH = './saved/models/esrgan.tflite'
IMG_PATH = "./images/input_hr/0855.png"
SAVE_IMG_PATH = "./images/results/tflite_0855.png"

def main():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    image = read_image(IMG_PATH)
    input_image = tf.image.resize(image, INPUT_SHAPE, method=tf.image.ResizeMethod.BICUBIC)
    input_image = scale_image_0_1_range(input_image)
    input_image = tf.expand_dims(input_image, axis=0)

    interpreter.set_tensor(input_index, input_image)

    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    # Convert output array to image
    output_image = (np.squeeze(output, axis=0).clip(0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(output_image)
    img.save(SAVE_IMG_PATH)











if __name__ == '__main__':
    main()