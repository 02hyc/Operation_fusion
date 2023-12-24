import tensorflow as tf
import numpy as np


def differential_testing(model, tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # 差分测试
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    # TensorFlow 预测
    tf_pred = model.predict(input_data)

    # TensorFlow Lite 预测
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_index)

    # 比较结果
    return np.allclose(tf_pred, tflite_pred, atol=1e-05)