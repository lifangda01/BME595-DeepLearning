# BME595-HW04
Neural Networks - Back-propagation in Torch.

# Network Implementation with GPU
In this homework, training on GPU is implemented by moving the criterion, network module and training/test data to the device memory of GPU.
For example:
```lua
local loss = nn.MSECriterion():cuda()
net:cuda()
trainImgs = trainImgs:cuda()
trainLabels = trainLabels:cuda()
testImgs = testImgs:cuda()
testLabels =testLabels:cuda()
```
# Performance Comparison
Here I evaluate the speed of training in terms of average time per epoch for three methods: `NeuralNetwork` in HW03, `nn` with CPU and `nn` with GPU. In particular, I did a simple benchmark on the training time and the test accuracy of the methods with different number of hidden neurons and different sizes of mini-batch. Note for this project, I'm using i7 4790k with 16GB RAM and GTX 980 TI. 

![alt text][time]
[time]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW04/time.png "Average Time (s) per Epoch"

![alt text][accu]
[accu]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW04/accu.png "Test Accuracy (%)"

It can be observed that in terms of training time, `nn` with CPU is the fasted. It's understandable that it outperforms our own `NeuralNetwork` module in every test case because of optimization reasons. However, not much speed up is shown by using GPU, which is counter-intuitive. Although it showed a small improvement of using GPU in the third test case, a much larger speed up was expected. One possible reason might be the scale. To not let the data transfer latency between RAM and GPU device memory become the bottleneck, we should use GPU for much larger scale of network and data than MNIST. Another possible reason might be my implementation details. Although I was careful to not allocate any new memory in GPU within my training loop and move all the relevant module and tensors to GPU, choices of parameters such as the mini-batch size might make a big difference.
It's surprising that our home-made `NeuralNetwork` module achieves far better accuracy than `nn`, especially with exactly same network structure and learning rate. This is very likely caused by the difference between the way we initialize our weights in `NeuralNetwork` and the way in `nn`. I believe after some tuning, `nn` should also achieve an accuracy as high as our `NeuralNetwork` module. Also note that there is no significant difference between `nn` with CPU and GPU, which is as expected.
