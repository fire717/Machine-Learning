from tensorflow.lite.python import schema_py_generated as schema_fb
import flatbuffers
import tensorflow as tf
import time
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0
 
#参考了https://github.com/raymond-li/tflite_tensor_outputter/blob/master/tflite_tensor_outputter.py
#调整output到指定idx
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    
    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)
    
    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]


# Read the model.
with open('lite-model_movenet_singlepose_lightning_3.tflite', 'rb') as f:
    model_buffer = f.read()
 
# 修改输出idx
idx = 95  #可以通过interpreter.get_tensor_details()，查各层的idx值
model_buffer = buffer_change_output_tensor_to(model_buffer, idx)
 
 
# 推理
interpreter = tf.lite.Interpreter(model_content=model_buffer)
interpreter.allocate_tensors()

print(interpreter.get_tensor_details())
 
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


image_path = '320240.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.image.resize_with_pad(image, 192, 192)
input_data = tf.cast(image, dtype=tf.float32)


interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
 
# 中间层的output值
out_val = interpreter.get_tensor(output_index)
print(out_val.shape)
