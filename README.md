# 软件测试代码作业报告

## 技术路线

在一开始拿到这个题目的时候我们不清楚到底要做什么，于是在网上进行检索，主要参考了https://zhuanlan.zhihu.com/p/664070841 和 https://www.tensorflow.org/lite/models/convert/operation_fusion （作业要求中给的这个链接似乎打不开 https://www.mindspore.cn/tutorial/zh-CN/r0.5/advanced_use/graph_kernel_fusion.html），最终我们决定使用tensorflow作为被测的深度学习框架。

在tensorflow文档的帮助下，我们确定了技术路线：首先确定一个复合算子，以Convolution + Bias Add（将卷积操作和偏置加法合并）为例。接着创建原始 TensorFlow 模型，这个模型包含一个卷积层，后面跟着一个偏置加法操作。

```python
def create_Conv2DBiasAdd_model():
    print("create_Conv2DBiasAdd_model")
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = Conv2DBiasAdd(32, (3, 3))(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```

使用 TensorFlow Lite Converter 将模型转换为 TensorFlow Lite 格式。转换过程中，Convolution + Bias Add"自动融合。

```python
def convert_to_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    return tflite_model
```

最后进行差分测试，使用相同的输入数据对两个模型进行预测，并比较输出结果，以检查它们之间的差异。

```python
# TensorFlow 预测
tf_pred = tf_model.predict(input_data[i:i + 1])
tf_preds.append(tf_pred)

# TensorFlow Lite 预测
interpreter.set_tensor(input_index, input_data[i:i + 1])
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_index)
tflite_preds.append(tflite_pred)
```



## 融合算子列表

我们梳理了Tensorflow框架下尽可能完整的融合算子列表：

**1.Convolution + Bias Add**

构建 TensorFlow 模型时，卷积层后紧跟着偏置项的加法。在转换为 TensorFlow Lite 时，使用 `tf.lite.Optimize.DEFAULT` 可以帮助将这两个操作自动融合。

**2.Batch Normalization**

TensorFlow Lite 通常会在转换过程中自动将批量标准化层与前面的卷积或全连接层融合。使用 `tf.keras.layers.BatchNormalization()` 构建批量标准化层，并确保它紧跟在卷积或全连接层后面。

**3.Relu6**

使用 `tf.nn.relu6` 作为激活函数，TensorFlow Lite 转换器会自动处理其融合。

**4.Depthwise Convolution + Pointwise Convolution**

使用 `tf.keras.layers.SeparableConv2D` 来实现可分离卷积，该层自动组合了深度卷积和逐点卷积。TensorFlow Lite 在转换过程中通常会保持这种融合结构。

**5.Sigmoid + Cross Entropy Loss**

在训练模型时，使用 `tf.keras.losses.BinaryCrossentropy(from_logits=True)` 并在输出层之前使用线性激活，将在内部进行融合优化。

**6.ReLU or Leaky ReLU + Max Pooling**

构建层时，将 ReLU 或 Leaky ReLU 激活函数和最大池化层相继排列。在许多情况下，TensorFlow Lite 会优化这些操作。

**7.MatMul + Add**

使用 `tf.keras.layers.Dense` 层，它内部实现了矩阵乘法和加法操作的融合。

**8.BatchNorm + ReLU**

在层的顺序排列中，将批量标准化层后跟 ReLU 激活层。这种组合通常会在 TensorFlow Lite 转换过程中自动融合。

**9.Conv2D + BatchNorm + ReLU**

这种组合可以通过将 `tf.keras.layers.Conv2D`、`tf.keras.layers.BatchNormalization` 和 ReLU 激活函数顺序排列实现。TensorFlow Lite 转换器通常能够自动识别和融合这种组合。

**10.GRU/LSTM operations**

使用 `tf.keras.layers.LSTM` 或 `tf.keras.layers.GRU`。TensorFlow Lite 转换器会优化这些循环层的操作，但具体的融合效果可能取决于模型的其他方面和转换时的设置。



## 实现 5 组融合算子并进行差分测试

代码仓库 https://github.com/02hyc/Operation_fusion.git

我们实现了以下 5 组融合算子

- Convolution + Bias Add
- Batch Normalization
- Depthwise Convolution + Pointwise Convolution
- Sigmoid + Cross Entropy Loss
- ReLU or Leaky ReLU + Max Pooling

在 `fusion_operators.py` 中定义了Tensorflow中的这五个复合运算。

在 `models.py` 中，分别生成了五个模型，并转换为 TensorFlow Lite。

在 `differential_testing.py` 中进行了差分测试

`main.py` 集合了上述流程，并将`两个模型的预测结果是否一致`输出到控制台

`saved_model_one` 到 `saved_model_five` 分别存储了模型一到五，使用 `saved_model_cli show --dir .\saved_model_one\ --all` 命令可以查看。