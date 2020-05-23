
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


data_path = r"/home/AlgorithmicGroup/yw/workshop/antiface/data/clean_data"
# 这个path指图像数据文件夹路径，其下面按类别分为多个子文件夹
data = ImageClassifierDataLoader.from_folder(data_path)
train_data, test_data = data.split(0.92)

print("done data load.")

model = image_classifier.create(train_data, 
  
                                model_spec=ms.efficientnet_lite0_spec,
                                shuffle=True,
                                validation_data=test_data,
                                batch_size=32,
                                epochs=20,
                                train_whole_model=False,
                                dropout_rate=0.2,
                                learning_rate=0.005,
                                momentum=0.9)
#指定模型为efficientnet_lite0，可以换成其他的
"""
def get_default_hparams():
  return HParams(
      train_epochs=5,
      do_fine_tuning=False,(train_whole_model)
      batch_size=32,
      learning_rate=0.005,
      momentum=0.9,
      dropout_rate=0.2)
"""


#loss, accuracy = model.evaluate(test_data)
#训练过程也会打印相关信息，类似keras


model.export('image_classifier.tflite', 'image_labels.txt')
#导出tflite模型，image_labels即对应的类别
