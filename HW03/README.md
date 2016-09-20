# BME595-HW03
Neural Networks - Back-propagation Pass.

## Logic Gates
Apparently, the weights obtained by back-propagation were much more modest in terms of magnitude. For example, for the XOR gate, the weights I chose were
```
torch.Tensor({{-10, 20, -20}, {-10, -20, 20}})
torch.Tensor({-10, 20, 20})
```
while the weights I got by back-propagation were
```
 2.6427 -6.6033 -6.6219
-7.5078  4.8291  4.8327
[torch.DoubleTensor of size 2x3]

 5.0544 -10.0531 -10.3876
[torch.DoubleTensor of size 1x3]
```
Nevertheless, both sets of weights were able to emulate the XOR gate.
## MNIST
The following results were obtained on the 10,000 test images, with `nLayers = 3` (1 hidden layer), `batchSize = 10` and `nEpoch = 50`. The classification rate below is chosen as the best I could get among all the epochs for a given network. A trained 100-neuron network is stored in `trainedNetwork.asc`. One can call `loadNN()` (currently local to the file as requested) to directly load the network and use it for classification.

 - 94.07%: `eta = 0.02`, `nNeuron = 30`;
 - 95.86%: `eta = 0.02`, `nNeuron = 50`;
 - 96.44%: `eta = 0.02`, `nNeuron = 100`;
 - 94.67%: `eta = 0.05`, `nNeuron = 30`;
 - 95.70%: `eta = 0.05`, `nNeuron = 50`;
 - 96.92%: `eta = 0.05`, `nNeuron = 100`.

As one can observe, the more neurons in the hidden layer, the better classification rate one could get. However, more neurons also significantly impact the training time. But it only takes about 10 minutes for the `nNeuron = 100` network to run 50 epochs.
The learning rate `eta` is also a key parameter. A reasonable higher `eta` apparently leads to convergence faster. Also, it seems higher `eta` also improves classification rate in general.
For future work, I can use adaptive learning rate, where I change my `eta` based on the current performance.