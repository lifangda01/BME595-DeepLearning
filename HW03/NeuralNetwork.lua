require 'math'
require 'torch'

-- Table stores key-value pairs, just like a dictionary in Python
local NeuralNetwork = {}
local Theta = {}
local dE_dTheta = {}
local a = {}
local z = {}

-- Prepend ones separately for 1D and 2D tensor
local function prependOnes(input)
	-- 2D
	if #input:size() == 2 then
		return torch.cat(torch.ones(1,input:size(2)), input, 1)
	-- 1D
	else
		return torch.cat(torch.ones(1), input)
	end
end

--- Mean Square Error
local function MSE(input, target)
	local d2 = torch.pow(input-target, 2) / 2.0
	return torch.sum(d2) / d2:view(-1,1):size(1)
end

-- create the table of matrices Θ
function NeuralNetwork.build(layer_sizes)
	Theta = {}
	dE_dTheta = {}
	a = {}
	z = {}
	if #layer_sizes < 2 then
		print("ERROR: must have at least 2 layers!")
		return
	end
	for i=1,#layer_sizes-1 do
		-- Don't forget the bias unit
		Theta[i] = torch.Tensor(layer_sizes[i+1], layer_sizes[i]+1):
					apply(	function () 
								return torch.normal(0, 1/math.sqrt(layer_sizes[i])) 
							end)
	end
end

-- returns Θ(layer)
function NeuralNetwork.getLayer(layer)
	return Theta[layer]
end

-- feedforward pass single vector
-- feedforward pass transposed design matrix (should come free after transposition)
function NeuralNetwork.forward(input)
	-- If input is a vector, we assume it's a column vector
	if #input:size() == 1 then
		input = input:view(-1,1)
	end
	-- Check for input dimension (with bias)
	if input:size(1)+1 ~= Theta[1]:size(2) then
		print("ERROR: input and theta dimension mismatch!")
		return
	end
	-- The activation of first layer is the input itself and bias
	-- a[1] = prependOnes(input:clone())
	a[1] = input:clone()
	-- Don't forget the bias unit
	local result
	for i,theta in ipairs(Theta) do
		if i == 1 then
			result = theta * prependOnes(input)
		else
			result = theta * prependOnes(result)
		end
		-- Store the weighted input z
		z[i+1] = result:clone()
		result:sigmoid()
		-- Store the activation a
		-- Activation of the bias unit is 1, also the last layer doesn't have bias
		-- if i == #Theta then
		-- 	a[i+1] = result:clone()
		-- else
		-- 	a[i+1] = prependOnes(result:clone())
		-- end
		a[i+1] = result:clone()
	end	
	return result
end

-- back-propagation pass single target (computes ∂E/∂Θ)
-- back-propagation pass target matrix (should come free after transp.)
-- (computes the average of ∂E/∂Θ across seen samples)
function NeuralNetwork.backward(target, loss)
	-- For every layer, we need to populate the dE_dTheta table, then we can use it for SGD
	-- In order to get dE_dTheta, we need to get error at every layer
	-- So we should start from the last layer
	-- To begin, let's calculate the error first
	-- print('a', a)
	-- print('Theta', Theta)
	-- print('nLayers', #Theta + 1)
	local nLayers = #Theta + 1
	local delta
	for i=nLayers,1,-1 do
		-- Compute the delta error at layer i
		if i == nLayers then
			-- print('i', i)
			-- print('a[i]', a[i])
			delta = (a[i] - target):cmul(a[i]):cmul(1 - a[i])
			-- print('delta', delta)
		elseif i == nLayers-1 then
			-- print('i', i)
			-- print('a[i]', a[i])
			-- print('Theta[i]', Theta[i])
			dE_dTheta[i] = ( prependOnes(a[i]) * delta:t() ):t()
			-- print('dE_dTheta[i]', dE_dTheta[i])
			delta = ( Theta[i]:t() * delta ) : cmul( prependOnes(a[i]) ) : cmul( 1 - prependOnes(a[i]) )
			-- print('delta', delta)
		else
			-- print('i', i)
			-- print('a[i]', a[i])
			-- print('Theta[i]', Theta[i])
			dE_dTheta[i] = ( prependOnes(a[i]) * delta:sub(2, delta:size(1)):t() ):t()
			-- print('dE_dTheta[i]', dE_dTheta[i])
			delta = ( Theta[i]:t() * delta:sub(2, delta:size(1)) ) : cmul( prependOnes(a[i]) ) : cmul( 1 - prependOnes(a[i]) )
			-- print('delta', delta)
		end
	end
	-- print("Error:", MSE(a[nLayers], target))
	-- FIXME: should return NOTHING!!!
	return MSE(a[nLayers], target)
end

-- update parameters
function NeuralNetwork.updateParams(etha)
	for i=1,#Theta do
		Theta[i] = Theta[i] - etha*dE_dTheta[i]
	end
end

return NeuralNetwork