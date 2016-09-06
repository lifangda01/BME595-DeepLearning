require 'math'
require 'torch'

-- Table stores key-value pairs, just like a dictionary in Python
local NeuralNetwork = {}
local network = {}

-- Prepend ones separately for 1D and 2D tensor
local function prepend_ones(input)
	-- 2D
	if #input:size() == 2 then
		return torch.cat(torch.ones(1,input:size(2)), input, 1)
	-- 1D
	else
		return torch.cat(torch.ones(1), input)
	end
end

-- create the table of matrices Θ
function NeuralNetwork.build(layer_sizes)
	network = {}
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
-- feedforward pass transposed design matrix (should come free)
function NeuralNetwork.forward(input)
	-- FIXME: ERROR CHECKING
	-- Don't forget the bias unit
	local result
	for i,theta in ipairs(network) do
		if i == 1 then
			result = theta * prepend_ones(input)
		else
			result = theta * prepend_ones(result)
		end
		result:sigmoid()
	end	
	return result
end

local function debug()
	NeuralNetwork.build({4,3,2,1})
	for i=1,#network do
		print(NeuralNetwork.getLayer(i))
	end
	NeuralNetwork.forward(torch.Tensor(4,4):zero())
end

-- debug()

return NeuralNetwork