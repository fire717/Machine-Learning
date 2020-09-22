# Pytorch Things


* labelsmooth with CrossEntropyLoss 
```
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
```

### Practice
* Pytorch 入门示例：
	1. [Pytorch是什么？](./practice/60分钟入门PyTorch-1.PyTorch是什么？.ipynb)
	2. [AUTOGRAD](./practice/60分钟入门PyTorch-2.AUTOGRAD.ipynb)
	3. [神经网络](./practice/60分钟入门PyTorch-3.神经网络.ipynb)
	4. [训练一个分类器](./practice/60分钟入门PyTorch-4.训练一个分类器.ipynb)
	5. [数据并行](./practice/60分钟入门PyTorch-5.数据并行.ipynb)

	> Note:This is part of this repo [Data-Science-Notes](https://github.com/fengdu78/Data-Science-Notes/tree/master/8.deep-learning/PyTorch_beginner).

* [NN example](./practice/pytorch_example.ipynb)
* [LSTM词性标注简单示例](./practice/pytorch_lstm.ipynb)
* [GAN示例？](./practice/gan_pytorch.py)
* [mnist demo](./practice/mnist_demo.py)

### Demo
* [可视化增强图片结果](./demo/show_pth_data.py)

### Resource
* [官网](http://pytorch.org/)
* [官网教程](http://pytorch.org/tutorials/)
* [PyTorch Handbook](https://github.com/zergtant/pytorch-handbook)
* Model zoo:[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) | [
pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 
