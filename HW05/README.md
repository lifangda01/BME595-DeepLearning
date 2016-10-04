# BME595-HW04
Neural Networks - CNN in Torch.

# `MNIST`
Using my old `nn` MLP model from the previous homework, the best result I was able to get on the testing set was,
 - 93.44%: `eta = 0.05`, `nNeuron = 30`, `batchSize = 10`.
The time it takes to train per epoch was 7.05 seconds on the CPU and it took about 20 epochs to converge to the 90% accuracy. The inference time for forwarding all the 10000 test images is about 0.45 seconds.

However, with the LeNet-5 model, along with the `optim` package, I was able to get:
 - 97.65%: `eta = 0.01`, `batchSize = 10`.
As a more complicated network than the simple MLP model, our LeNet-5 network indeed achieves higher accuracy with the price of training speed. Each epoch took 13.61 seconds on the CPU and about 17 epochs are needed to pass the 95% accuracy. Also, the inference time for forwarding all the 10000 test images is more than 2 seconds, which is as expected due to the convolutions and larger number of hidden layers and hidden neurons.


# `CIFAR`
For the `cifar-10` dataset, I was able to achieve
- 54.25%: `eta = 0.1`, `batchSize = 10`.
The training time is significantly greater than `MNIST`, with each epoch of training the 50000 32x32 RGB images taking about 24.8 seconds. 

For the `cifar-100` dataset, my LeNet-5 can only achieve an accuracy of 23.73%. The conclusion is that LeNet-5 is too simple for classifying a 100-class dataset. 
# Visualization
With my `cifar-10` network, I was able to classify objects based on my continuous camera feed. Here is a frame from my webcam of me holding a picture of an airplane, with its prediction on the top left (note that the number label for planes is 0 in `cifar-10`).

![alt text][webcam]
[webcam]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW05/vis.png "Webcam feed"

Here I also show two correct classification results for images in the test set:

![alt text][cat]
[cat]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW05/cat.png "Cat"
![alt text][ship]
[ship]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW05/ship.png "Ship"

Here is an misclassified image, where a plane is misclassified as an automobile:

![alt text][plane]
[plane]: https://github.com/lifangda01/BME595-DeepLearning/blob/master/HW05/wrong.png "Plane"
