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