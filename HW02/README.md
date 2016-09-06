# BME595-HW02
Neural Networks - Feedforward Pass.
## API (`NeuralNetwork`)
```
-- create the table of matrices Θ
[nil] build({[int] in, [int] h1, [int] h2, …, [int] out})
-- returns Θ(layer)
[2D DoubleTensor] getLayer([int] layer)
-- feedforward pass single vector
[1D DoubleTensor] forward([1D DoubleTensor] input)
-- feedforward pass transposed design matrix (should come free)
[2D DoubleTensor] forward([2D DoubleTensor] input)
```
## Secondary API (`logicGates`)
```
[boolean] AND([boolean] x, [boolean] y)
[boolean]  OR([boolean] x, [boolean] y)
[boolean] NOT([boolean] x)
[boolean] XOR([boolean] x, [boolean] y)
```
## Sample Usage
### API
```
local NN = require 'NeuralNetwork'
```
Initiate the network.
```
NN.build({4,3,2,1})
```
Modify the weights in layer 1 for the first element in layer 2.
```
NN.getLayer(1)[1] = torch.Tensor({10, -20, -30, 40})
```
Feed forward pass.
```
NN.forward(torch.Tensor(4,4):ones())
```
### Secondary API
Test the functionality of the gates.
```
print("-----------Testing NOTXOR-----------")
print( (logicGates.NOT(logicGates.XOR(false,false)) == true) and "Passed" or "Failed")
print( (logicGates.NOT(logicGates.XOR(false,true)) == false) and "Passed" or "Failed")
print( (logicGates.NOT(logicGates.XOR(true,false)) == false) and "Passed" or "Failed")
print( (logicGates.NOT(logicGates.XOR(true,true)) == true) and "Passed" or "Failed")
```
##Class Version
Currently the neural network is implemented using a table. Its OOP version is to be worked on.