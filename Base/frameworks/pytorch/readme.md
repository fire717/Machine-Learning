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

* 梯度裁剪
```
def clip_gradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
```

* 导出onnx
```
def test():
    model = SiameseNetworkOutput()
    pthfile = os.path.join(Config.save_dir,'3e47_loss0.23540_acc0.96000.pth')
    model.load_state_dict(torch.load(pthfile, map_location='cpu'), strict=True)
    #data type nchw
    dummy_input1 = torch.randn(1, 3, 100, 100)
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "output/mymodel.onnx", verbose=True, input_names=input_names, output_names=output_names)
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
* [onnx predict](./demo/onnx_pre.py)

### Toolkit
* [Focal-Loss](https://github.com/yatengLG/Focal-Loss-Pytorch)
* [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
* [warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
* [fast-autoaugment](https://github.com/kakaobrain/fast-autoaugment)
* [各种IOUloss的pytorch实现](toolkit/IoU_loss.py)

### Resource
* [官网](http://pytorch.org/)
* [官网教程](http://pytorch.org/tutorials/)
* [PyTorch Handbook](https://github.com/zergtant/pytorch-handbook)
* [pytorch_classification](https://github.com/lxztju/pytorch_classification)
* Model zoo:
	* [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) 
	* [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 
	* [mobilenet-v3](https://github.com/kuan-wang/pytorch-mobilenet-v3) 
	* [WSL-Images](https://github.com/facebookresearch/WSL-Images)
	
	
