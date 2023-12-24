import numpy as np
import fusion_operators as md
import models as mc
import differential_testing as dt
import tensorflow as tf


model_list = ["saved_model_one", "saved_model_two", "saved_model_three", "saved_model_four"]
model_creation_methods = [getattr(mc, m) for m in dir(mc) if callable(getattr(mc, m)) and m.startswith('create_')]


for i in range(len(model_list)):
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
    for i in range(len(tf_pred)):
        print(f"{i+1} - TensorFlow: {tf_pred[i]}, TensorFlow Lite: {tflite_pred[i]}")

    # 判断两个模型的输出是否一致
    consistent = np.allclose(tf_pred, tflite_pred, atol=1e-05)
    if consistent:
        print("两个模型的预测结果一致。")
    else:
        print("两个模型的预测结果不一致。")
