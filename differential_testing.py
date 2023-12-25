import tensorflow as tf
import numpy as np


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
