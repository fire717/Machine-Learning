
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


data_path = "D:/Data/clothes_style/data/DeepFashion/classes7/data_val"
# 这个path指图像数据文件夹路径，其下面按类别分为多个子文件夹
data = ImageClassifierDataLoader.from_folder(data_path)
train_data, test_data = data.split(0.9)

print("done data load.")

model = image_classifier.create(train_data, model_spec=ms.efficientnet_lite0_spec)
#指定模型为efficientnet_lite0，可以换成其他的


loss, accuracy = model.evaluate(test_data)
#训练过程也会打印相关信息，类似keras


model.export('image_classifier.tflite', 'image_labels.txt')
#导出tflite模型，image_labels即对应的类别
