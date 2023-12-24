from fusion_operators import Conv2DBiasAdd, Conv2DBatchNorm, DepthwisePointwiseConv, SigmoidCrossEntropyModel
import tensorflow as tf
import numpy as np


def create_Conv2DBiasAdd_model():
    print("create_Conv2DBiasAdd_model")
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = Conv2DBiasAdd(32, (3, 3))(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_batch_norm_model():
    print("create_batch_norm_model")
    model = tf.keras.models.Sequential([
        Conv2DBatchNorm(32, (3, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model


def create_depthwise_pointwise_model():
    print("create_depthwise_pointwise_model")
    model = tf.keras.models.Sequential([
        DepthwisePointwiseConv(32, (3, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    return model


def create_sigmoid_crossentropy_model():
    model = SigmoidCrossEntropyModel()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    model.predict(np.random.rand(1, 28, 28, 1))
    return model



def convert_to_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    return tflite_model


