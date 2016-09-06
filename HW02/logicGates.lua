require 'math'
require 'torch'

local NN = require 'NeuralNetwork'
local logicGates = {}
local epsilon = 1e-4

-- Simple helper functions for single value type conversion
local function B2I(b)
	return b and 1 or 0
end

local function I2B(i)
	if math.abs(i-1) < epsilon then
		return true
	end
	return false
end

function logicGates.NOT(x)
	NN.build({1,1})
	NN.getLayer(1)[1] = torch.Tensor({10, -20})
	return I2B( NN.forward(torch.Tensor({ B2I(x) }))[1] )
end

function logicGates.AND(x, y)
	NN.build({2,1})
	NN.getLayer(1)[1] = torch.Tensor({-30, 20, 20})
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) }))[1] )
end

function logicGates.OR(x, y)
	NN.build({2,1})
	NN.getLayer(1)[1] = torch.Tensor({-10, 20, 20})
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) }))[1] )
end

function logicGates.XOR(x, y)
	NN.build({2,2,1})
	NN.getLayer(1)[{{1,2}}] = torch.Tensor({{-10, 20, -20}, {-10, -20, 20}})
	NN.getLayer(2)[1] = torch.Tensor({-10, 20, 20})
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) }))[1] )
end

local function debug()
	print("-----------Testing NOT-----------")
	print( (logicGates.NOT(true) == false) and "Passed" or "Failed")
	print( (logicGates.NOT(false) == true) and "Passed" or "Failed")

	print("-----------Testing AND-----------")
	print( (logicGates.AND(false,false) == false) and "Passed" or "Failed")
	print( (logicGates.AND(false,true) == false) and "Passed" or "Failed")
	print( (logicGates.AND(true,false) == false) and "Passed" or "Failed")
	print( (logicGates.AND(true,true) == true) and "Passed" or "Failed")

	print("-----------Testing OR-----------")
	print( (logicGates.OR(false,false) == false) and "Passed" or "Failed")
	print( (logicGates.OR(false,true) == true) and "Passed" or "Failed")
	print( (logicGates.OR(true,false) == true) and "Passed" or "Failed")
	print( (logicGates.OR(true,true) == true) and "Passed" or "Failed")

	print("-----------Testing XOR-----------")
	print( (logicGates.XOR(false,false) == false) and "Passed" or "Failed")
	print( (logicGates.XOR(false,true) == true) and "Passed" or "Failed")
	print( (logicGates.XOR(true,false) == true) and "Passed" or "Failed")
	print( (logicGates.XOR(true,true) == false) and "Passed" or "Failed")

	print("-----------Testing NOTXOR-----------")
	print( (logicGates.NOT(logicGates.XOR(false,false)) == true) and "Passed" or "Failed")
	print( (logicGates.NOT(logicGates.XOR(false,true)) == false) and "Passed" or "Failed")
	print( (logicGates.NOT(logicGates.XOR(true,false)) == false) and "Passed" or "Failed")
	print( (logicGates.NOT(logicGates.XOR(true,true)) == true) and "Passed" or "Failed")

	print("-----------Testing ANDOR-----------")
	print( (logicGates.OR(logicGates.AND(false,false),true) == true) and "Passed" or "Failed")
	print( (logicGates.OR(logicGates.AND(false,true),false) == false) and "Passed" or "Failed")
	print( (logicGates.OR(logicGates.AND(true,false),false) == false) and "Passed" or "Failed")
	print( (logicGates.OR(logicGates.AND(true,true),false) == true) and "Passed" or "Failed")
end

debug()

return logicGates