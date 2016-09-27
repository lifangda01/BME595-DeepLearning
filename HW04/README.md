# BME595-HW04
Neural Networks - Back-propagation in Torch.

# `nn` with GPU
In this homework, training on GPU is implemented by moving the criterion, network module and training/test data to the device memory of GPU.
For example,
```
local loss = nn.MSECriterion():cuda()
net:cuda()
trainImgs = trainImgs:cuda()
trainLabels = trainLabels:cuda()
testImgs = testImgs:double():cuda()
testLabels = testset.label:cuda()
```
