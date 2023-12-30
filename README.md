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



## 分析代码

代码仓库 https://github.com/02hyc/Operation_fusion.git

我们实现了以下 **5 组融合算子并进行差分测试**

- Convolution + Bias Add
- Batch Normalization
- Depthwise Convolution + Pointwise Convolution
- Sigmoid + Cross Entropy Loss
- ReLU or Leaky ReLU + Max Pooling

### 1.fusion_operators.py

自定义融合运算：定义了五个使用TensorFlow进行神经网络操作的自定义层。

- Conv2DBiasAdd

  ```python
  class Conv2DBiasAdd(tf.keras.layers.Layer):
      def __init__(self, filters, kernel_size):
          super(Conv2DBiasAdd, self).__init__()
          self.conv = tf.keras.layers.Conv2D(filters, kernel_size)
          self.bias = self.add_weight("bias", shape=(filters,))
  
      @tf.function(experimental_implements="Conv2DBiasAdd")
      def call(self, inputs):
          x = self.conv(inputs)
          return x + self.bias
  ```

  - 创建了一个卷积层 `self.conv`，使用 `tf.keras.layers.Conv2D`，其中包含了 `filters` 个滤波器和指定大小的卷积核。
  - 使用 `self.add_weight` 方法添加一个名为 "bias" 的可训练参数，其形状为 `(filters,)`，即每个滤波器对应一个偏置。
  - `call`方法中将卷积结果与添加的偏置 `self.bias` 相加，实现了卷积后添加偏置的操作

- Conv2DBatchNorm

  ```python
  class Conv2DBatchNorm(tf.keras.layers.Layer):
      def __init__(self, filters, kernel_size):
          super(Conv2DBatchNorm, self).__init__()
          self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
          self.batch_norm = tf.keras.layers.BatchNormalization()
  
      @tf.function(experimental_implements="Conv2DBatchNorm")
      def call(self, inputs, training=False):
          x = self.conv(inputs)
          return self.batch_norm(x, training=training)
  ```

  - 创建了一个卷积层 `self.conv`，使用 `tf.keras.layers.Conv2D`，其中包含了 `filters` 个滤波器、指定大小的卷积核，并设置 `padding='same'` 表示使用相同的填充。

  - 创建了一个批量归一化层 `self.batch_norm`，使用 `tf.keras.layers.BatchNormalization()`。

  - `call` 方法将卷积结果传递给批量归一化层 `self.batch_norm` 进行批量归一化操作，同时传递训练模式参数 `training`。

- DepthwisePointwiseConv

  ```python
  class DepthwisePointwiseConv(tf.keras.layers.Layer):
      def __init__(self, filters, kernel_size):
          super(DepthwisePointwiseConv, self).__init__()
          self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same')
          self.pointwise_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
  
      @tf.function(experimental_implements="DepthwisePointwiseConv")
      def call(self, inputs):
          x = self.depthwise_conv(inputs)
          return self.pointwise_conv(x)
  ```

  - 创建了一个深度可分离卷积层 `self.depthwise_conv`，使用 `tf.keras.layers.DepthwiseConv2D`，其中包含了指定大小的卷积核，设置 `padding='same'` 表示使用相同的填充。
  - 创建了一个普通卷积层 `self.pointwise_conv`，使用 `tf.keras.layers.Conv2D`，其中包含了 `filters` 个滤波器、大小为 (1, 1) 的卷积核，同样设置 `padding='same'`。
  - `call` 方法将输入传递给深度可分离卷积层 `self.depthwise_conv` 进行深度可分离卷积操作，其结果传递给普通卷积层 `self.pointwise_conv` 进行点-wise（逐点）卷积操作。

- SigmoidCrossEntropyModel

  ```python
  class SigmoidCrossEntropyModel(tf.keras.Model):
      def __init__(self):
          super(SigmoidCrossEntropyModel, self).__init__()
          self.input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
          self.dense = tf.keras.layers.Dense(1, activation='linear')
  
      @tf.function(experimental_implements="SigmoidCrossEntropy")
      def call(self, inputs):
          x = self.input_layer(inputs)
          return self.dense(x)
  ```

  - 创建了一个输入层 `self.input_layer`，使用 `tf.keras.layers.InputLayer`，指定输入形状为 (28, 28, 1)。
  - 创建了一个全连接层 `self.dense`，使用 `tf.keras.layers.Dense`，输出维度为 1，激活函数为线性激活函数。
  - `call` 方法将输入传递给输入层 `self.input_layer` 进行处理，处理后的结果传递给全连接层 `self.dense` 进行线性激活函数的全连接操作。

- LeakyReLU_MaxPooling

  ```python
  class LeakyReLU_MaxPooling(tf.keras.layers.Layer):
      def __init__(self):
          super(LeakyReLU_MaxPooling, self).__init__()
          self.leaky_relu = tf.keras.layers.LeakyReLU()
          self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
  
      @tf.function(experimental_implements="LeakyReLU_MaxPooling")
      def call(self, inputs):
          x = self.leaky_relu(inputs)
          return self.max_pool(x)
  ```

  - 创建了一个 Leaky ReLU 层 `self.leaky_relu`，使用 `tf.keras.layers.LeakyReLU` 默认配置。
  - 创建了一个最大池化层 `self.max_pool`，使用 `tf.keras.layers.MaxPooling2D`，设置池化窗口大小为 (2, 2)。
  - `call` 方法将输入传递给 Leaky ReLU 层 `self.leaky_relu` 进行 Leaky ReLU 操作，其结果传递给最大池化层 `self.max_pool` 进行最大池化操作。



### 2. models.py

使用自定义融合运算符创建不同神经网络模型。

#### 2.1 模型创建函数

- create_Conv2DBiasAdd_model

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

  - 使用 `tf.keras.Input` 定义输入层，指定输入形状为 (28, 28, 1)。
  - 将输入通过 `Conv2DBiasAdd` 层进行处理，该层是之前定义的自定义层，具有卷积后添加偏置的功能。
  - 使用 `tf.keras.layers.Flatten()` 将卷积操作后的结果展平。
  - 将展平后的结果通过全连接层 `tf.keras.layers.Dense(10)` 进行处理，输出维度为 10。
  - 创建 `tf.keras.Model`，指定输入和输出，形成完整的模型。

- create_batch_norm_model

  ```python
  def create_batch_norm_model():
      print("create_batch_norm_model")
      model = tf.keras.models.Sequential([
          Conv2DBatchNorm(32, (3, 3)),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(10)
      ])
      return model
  ```

  - 使用 `tf.keras.models.Sequential` 创建一个序列模型，该模型由一系列层按顺序堆叠而成。
  - 将 `Conv2DBatchNorm` 自定义层添加到模型中，设置滤波器数量为 32，卷积核大小为 (3, 3)。
  - 添加 ReLU 激活函数层。
  - 添加展平层，将卷积操作后的结果展平。
  - 添加一个全连接层 `tf.keras.layers.Dense(10)`，输出维度为 10。

- create_depthwise_pointwise_model

  ```python
  def create_depthwise_pointwise_model():
      print("create_depthwise_pointwise_model")
      model = tf.keras.models.Sequential([
          DepthwisePointwiseConv(32, (3, 3)),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(10)
      ])
      return model
  ```

  - 使用 `tf.keras.models.Sequential` 创建一个序列模型，该模型由一系列层按顺序堆叠而成。
  - 将 `DepthwisePointwiseConv` 自定义层添加到模型中，设置滤波器数量为 32，卷积核大小为 (3, 3)。
  - 添加 ReLU 激活函数层。
  - 添加展平层，将深度可分离卷积操作后的结果展平。
  - 添加一个全连接层 `tf.keras.layers.Dense(10)`，输出维度为 10。

- create_sigmoid_crossentropy_model

  ```python
  def create_sigmoid_crossentropy_model():
      model = SigmoidCrossEntropyModel()
      model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
      model.predict(np.random.rand(1, 28, 28, 1))
      return model
  ```

  - 创建了一个 `SigmoidCrossEntropyModel` 自定义模型实例，该模型包含一个输入层、一个全连接层，激活函数为线性激活函数。
  - 使用 `model.compile` 配置模型的优化器为 Adam，并指定损失函数为二元交叉熵（BinaryCrossentropy），设置 `from_logits=True` 表示模型输出为 logits。
  - 使用 `model.predict` 进行一次随机输入的预测，这一步是为了确保模型的权重被正确初始化。

- create_leaky_relu_max_pooling_model

  ```python
  def create_leaky_relu_max_pooling_model():
      model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
          LeakyReLU_MaxPooling(),
          tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')  # 假设是一个10类分类问题
      ])
      return model
  ```

  - 使用 `tf.keras.models.Sequential` 创建一个序列模型，该模型由一系列层按顺序堆叠而成。
  - 添加一个卷积层 `tf.keras.layers.Conv2D`，设置滤波器数量为 32，卷积核大小为 (3, 3)，激活函数为 ReLU，输入形状为 (28, 28, 1)。
  - 添加之前定义的 `LeakyReLU_MaxPooling` 自定义层。
  - 添加另一个卷积层 `tf.keras.layers.Conv2D`，设置滤波器数量为 64，卷积核大小为 (3, 3)，激活函数为 ReLU。
  - 添加一个展平层 `tf.keras.layers.Flatten()`。
  - 添加一个全连接层 `tf.keras.layers.Dense`，设置神经元数量为 128，激活函数为 ReLU。
  - 添加最后一个全连接层 `tf.keras.layers.Dense`，设置神经元数量为 10，激活函数为 softmax，假设是一个 10 类分类问题。

#### 2.2 TFLite转换函数

```python
def convert_to_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    return tflite_model
```

使用`tf.lite.TFLiteConverter`将保存的TensorFlow模型转换为TensorFlow Lite格式。



### 3. differential_testing.py

提供了一个用于在TensorFlow和TensorFlow Lite模型之间进行差分测试的函数。

```python
def differential_testing(tf_model, tflite_model_content, input_data):
    # 初始化 TensorFlow Lite 解释器
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # 逐个样本进行预测
    tf_preds = []
    tflite_preds = []
    for i in range(input_data.shape[0]):
        # TensorFlow 预测
        tf_pred = tf_model.predict(input_data[i:i + 1])
        tf_preds.append(tf_pred)

        # TensorFlow Lite 预测
        interpreter.set_tensor(input_index, input_data[i:i + 1])
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_index)
        tflite_preds.append(tflite_pred)

    # 返回所有样本的预测结果
    return tf_preds, tflite_preds
```

- 接受三个参数：
  - `tf_model`：TensorFlow 模型，通过该模型进行预测。
  - `tflite_model_content`：TensorFlow Lite 模型的内容，即经过转换后的二进制字符串。
  - `input_data`：输入数据，包含多个样本，形状为 `(num_samples, 28, 28, 1)`。
- 初始化 TensorFlow Lite 解释器 `interpreter`，加载模型内容，并分配张量，获取输入和输出张量的索引。
- 使用循环逐个样本进行预测比较：
  - 对于 TensorFlow 模型，使用 `tf_model.predict` 进行预测，并将结果添加到 `tf_preds` 列表中。
  - 对于 TensorFlow Lite 模型，使用解释器设置输入张量并调用 `interpreter.invoke()` 进行预测，然后获取输出张量的值，并将结果添加到 `tflite_preds` 列表中。
- 返回两个列表，分别包含 TensorFlow 模型和 TensorFlow Lite 模型在所有样本上的预测结果。



### 4. main.py

集合了上述流程，创建、保存、转换模型，并进行差分测试比较 TensorFlow 模型和 TensorFlow Lite 模型在生成的随机测试数据上的预测结果，并将`两个模型的预测结果是否一致`输出到控制台。

- 模型列表和创建

  ```python
  model_list = ["saved_model_one", "saved_model_two", "saved_model_three", "saved_model_four", "saved_model_five"]
  model_creation_methods = [getattr(mc, m) for m in dir(mc) if callable(getattr(mc, m)) and m.startswith('create_')]
  ```

  - `model_list` 包含要保存的模型名称。
  - `model_creation_methods` 是一个列表，其中包含了通过反射从 `models.py` 模块中获取的以 `'create_'` 开头的可调用对象，这些对象是用于创建模型的方法。

- 模型创建和测试循环

  ```python
  for i in range(len(model_list)):
      with open(f'log_{i + 1}.txt', 'w') as f:
          # 创建并保存 TensorFlow 模型
          model = model_creation_methods[i]()
          model.build(input_shape=(None, 28, 28, 1))
          tf.saved_model.save(model, model_list[i])
  
          # 转换模型为 TensorFlow Lite 格式
          tflite_model = mc.convert_to_tflite(model_list[i])
  
          # 生成测试数据
          test_data = np.random.rand(500, 28, 28, 1).astype(np.float32)
  
          # 执行差分测试
          tf_pred, tflite_pred = dt.differential_testing(model, tflite_model, test_data)
  
          for j in range(len(tf_pred)):
              line = f"{j + 1} - TensorFlow: {tf_pred[j]}, TensorFlow Lite: {tflite_pred[j]}\n"
              print(line)
              f.write(line)
  
          # 判断两个模型的输出是否一致
          consistent = np.allclose(tf_pred, tflite_pred, atol=1e-05)
          if consistent:
              result = "两个模型的预测结果一致。\n"
          else:
              result = "两个模型的预测结果不一致。\n"
          print(result)
          f.write(result)
  ```

  - 遍历`model_list`，对于每个模型：
    - 调用`model_creation_methods`中相应的模型创建方法，使用输入形状为(None, 28, 28, 1)创建TensorFlow模型，保存模型为TensorFlow SavedModel格式。
    - 使用 `mc.convert_to_tflite` 将保存的模型转换为 TensorFlow Lite 格式。
    - 为差分测试生成随机测试数据`test_data`，形状为 `(500, 28, 28, 1)`。
    - 使用`differential_testing`方法执行差分测试，获取 TensorFlow 模型和 TensorFlow Lite 模型在测试数据上的预测结果。
    - 判断每个样本的 TensorFlow 模型和 TensorFlow Lite 模型的预测结果是否一致，将结果输出到日志文件并打印到控制台。



`saved_model_one` 到 `saved_model_five` 分别存储了模型一到五，使用 `saved_model_cli show --dir .\saved_model_one\ --all` 命令可以查看。

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 28, 28, 1)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

Concrete Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
```













