import numpy as np
import fusion_operators as md
import models as mc
import differential_testing as dt
import tensorflow as tf

# 创建并保存 TensorFlow 模型
model = mc.create_Conv2DBiasAdd_model()
tf.saved_model.save(model, "saved_model_one")

# 转换模型为 TensorFlow Lite 格式
tflite_model = mc.convert_to_tflite("saved_model_one")

# 生成测试数据
test_data = np.random.rand(1, 28, 28, 1).astype(np.float32)

# 执行差分测试
result = dt.differential_testing(model, tflite_model, test_data)
print("两个模型的预测是否一致:", result)
