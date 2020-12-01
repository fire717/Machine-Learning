# load model and predicate
import mxnet as mx
import numpy as np
import cv2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def load_model(prefix, epoch, ctx, height, width):
    print(prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 3, int(height), int(width)))])
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
    return sym, mod

height, width = 112,112
load_epoch = 0
model_prefix = "mynet"
sym, mod = load_model(model_prefix, load_epoch, mx.cpu(), height, width)  # ctx = mx.cpu()  mx.gpu(0)

img=cv2.imread('./h.jpg')
img=cv2.resize(img,(width, height))
img = np.reshape(img, (3, height, width))
img = np.array([img])

print(img.shape)
img = mx.nd.array(img)
mod.forward(Batch([img]))     
print('height', height, 'width', width)
print('img',img[0,2,0])
prob = mod.get_outputs()[0].asnumpy()

print(prob.shape)
