require 'math'
require 'torch'

-- Table stores key-value pairs, just like a dictionary in Python
local NeuralNetwork = {}
local network = {}

-- create the table of matrices Θ
function NeuralNetwork.build(layer_sizes)
	for i=1,#layer_sizes-1 do
		-- Don't forget the bias unit
		network[i] = torch.Tensor(layer_sizes[i+1], layer_sizes[i]+1):
					apply(	function () 
								return torch.normal(0, 1/math.sqrt(layer_sizes[i])) 
							end)
	end
end

-- returns Θ(layer)
function NeuralNetwork.getLayer(layer)
	return network[layer]
end

-- feedforward pass single vector
function NeuralNetwork.forward()
	
end

-- feedforward pass transposed design matrix (should come free)
-- [2D DoubleTensor] forward([2D DoubleTensor] input)

function debug()
	-- NeuralNetwork.build({4,3,2,1})
	-- print(NeuralNetwork.getLayer(1))
	-- print(NeuralNetwork.getLayer(2))
	-- print(NeuralNetwork.getLayer(3))
	
end

debug()

return NeuralNetwork