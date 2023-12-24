from fusion_operators import Conv2DBiasAdd
import tensorflow as tf


def create_Conv2DBiasAdd_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = Conv2DBiasAdd(32, (3, 3))(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def convert_to_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    return tflite_model