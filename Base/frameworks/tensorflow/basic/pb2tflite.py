import tensorflow as tf
 
import pathlib2 as pathlib
 
 
# 1.伪量化
# converter = tf.contrib.lite.TocoConverter.from_frozen_graph('model.pb',["input_image"],["result"], input_shapes={"input_image":[1,626,361,3]})   #Python 2.7.6版本,但测试量化后模型大小不会变小
converter = tf.lite.TFLiteConverter.from_frozen_graph('frozen_insightface_r50.pb',["data"],["output"], input_shapes={"data":[1,112,112,3]})   #python3.4.3--nightly版本,测试量化后模型大小会变小
 
converter.post_training_quantize = True
 
tflite_quantized_model=converter.convert()
 
open("quantized_model.tflite", "wb").write(tflite_quantized_model)



# 2 量化
# converter = tf.lite.TFLiteConverter.from_frozen_graph('frozen_insightface_r50.pb',["data"],["output"], input_shapes={"data":[1,112,112,3]})   #python3.4.3--nightly版本,测试量化后模型大小会变小
 
# converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
 
# converter.quantized_input_stats = {"data" : (127, 2.)}
 
# converter.default_ranges_stats=(0, 6)
 
# tflite_quantized_model=converter.convert()
 
# open("true_quantized_model.tflite", "wb").write(tflite_quantized_model)
