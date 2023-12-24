import tensorflow as tf


# 定义复合运算
class Conv2DBiasAdd(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv2DBiasAdd, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size)
        self.bias = self.add_weight("bias", shape=(filters,))

    @tf.function(experimental_implements="Conv2DBiasAdd")
    def call(self, inputs):
        x = self.conv(inputs)
        return x + self.bias


class Conv2DBatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv2DBatchNorm, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()

    @tf.function(experimental_implements="Conv2DBatchNorm")
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        return self.batch_norm(x, training=training)


class DepthwisePointwiseConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(DepthwisePointwiseConv, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same')
        self.pointwise_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')

    @tf.function(experimental_implements="DepthwisePointwiseConv")
    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        return self.pointwise_conv(x)


class SigmoidCrossEntropyModel(tf.keras.Model):
    def __init__(self):
        super(SigmoidCrossEntropyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    @tf.function(experimental_implements="SigmoidCrossEntropy")
    def call(self, inputs):
        x = self.input_layer(inputs)
        return self.dense(x)