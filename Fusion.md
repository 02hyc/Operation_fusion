### 1. **Convolution + Bias Add**

- 确保在构建 TensorFlow 模型时，卷积层后紧跟着偏置项的加法。
- 在转换为 TensorFlow Lite 时，使用 `tf.lite.Optimize.DEFAULT` 可以帮助将这两个操作自动融合。

### 2. **Batch Normalization**

- TensorFlow Lite 通常会在转换过程中自动将批量标准化层与前面的卷积或全连接层融合。
- 使用 `tf.keras.layers.BatchNormalization()` 构建批量标准化层，并确保它紧跟在卷积或全连接层后面。

### 3. **Relu6**

- 使用 `tf.nn.relu6` 作为激活函数，TensorFlow Lite 转换器会自动处理其融合。

### 4. **Depthwise Convolution + Pointwise Convolution**

- 使用 `tf.keras.layers.SeparableConv2D` 来实现可分离卷积，该层自动组合了深度卷积和逐点卷积。
- TensorFlow Lite 在转换过程中通常会保持这种融合结构。

### 5. **Sigmoid + Cross Entropy Loss**

- 在训练模型时，使用 `tf.keras.losses.BinaryCrossentropy(from_logits=True)` 并在输出层之前使用线性激活，将在内部进行融合优化。

### 6. **ReLU or Leaky ReLU + Max Pooling**

- 构建层时，将 ReLU 或 Leaky ReLU 激活函数和最大池化层相继排列。在许多情况下，TensorFlow Lite 会优化这些操作。

### 7. **MatMul + Add**

- 使用 `tf.keras.layers.Dense` 层，它内部实现了矩阵乘法和加法操作的融合。

### 8. **BatchNorm + ReLU**

- 在层的顺序排列中，将批量标准化层后跟 ReLU 激活层。
- 这种组合通常会在 TensorFlow Lite 转换过程中自动融合。

### 9. **Conv2D + BatchNorm + ReLU**

- 这种组合可以通过将 `tf.keras.layers.Conv2D`、`tf.keras.layers.BatchNormalization` 和 ReLU 激活函数顺序排列实现。
- TensorFlow Lite 转换器通常能够自动识别和融合这种组合。

### 10. **GRU/LSTM operations**

- 使用 `tf.keras.layers.LSTM` 或 `tf.keras.layers.GRU`。
- TensorFlow Lite 转换器会优化这些循环层的操作，但具体的融合效果可能取决于模型的其他方面和转换时的设置。