require 'math'
require 'torch'

local gnuplot = require 'gnuplot'
local NN = require 'NeuralNetwork'
local NOT = {}
local AND = {}
local OR = {}
local XOR = {}
local logicGates = {NOT, AND, OR, XOR}
local epsilon = 1e-4

-- Simple helper functions for single value type conversion
local function B2I(b)
	return b and 1 or 0
end

local function I2B(i)
	if torch.round(i) == 1 then
		return true
	end
	return false
end

function NOT.set()
	NN.build({1,1})
	NN.getLayer(1)[1] = torch.Tensor({10, -20})
end

function NOT.train()
	local input, target
	local maxIter = 100000
	local E = torch.Tensor(maxIter)
	local etha = 1.0
	input = torch.Tensor({
		{0, 1}
		})
	target = torch.Tensor({
		{1, 0}
		})
	NN.build({1,1})
	for i=1,maxIter do
		NN.forward(input)
		local e = NN.backward(target, 'MSE')
		E[i] = e
		if e < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
		NN.updateParams(etha)
	end
	-- gnuplot.plot(E)
	-- print(NN.getLayer(1))
end

function NOT.forward(x)
	-- print( NN.forward(torch.Tensor({ B2I(x) })) : view(-1,1)[1][1] )
	return I2B( NN.forward(torch.Tensor({ B2I(x) })) : view(-1,1)[1][1] )
end

function AND.set()
	NN.build({2,1})
	NN.getLayer(1)[1] = torch.Tensor({-30, 20, 20})
end

function AND.train()
	local input, target
	local maxIter = 100000
	local E = torch.Tensor(maxIter)
	local etha = 1.0
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		})
	target = torch.Tensor({
		{0, 0, 0, 1}
		})
	NN.build({2,1})
	for i=1,maxIter do
		NN.forward(input)
		local e = NN.backward(target, 'MSE')
		E[i] = e
		if e < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
		NN.updateParams(etha)
	end
	-- gnuplot.plot(E)
	-- print(NN.getLayer(1))
end

function AND.forward(x, y)
	-- print( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
end

function OR.set()
	NN.build({2,1})
	NN.getLayer(1)[1] = torch.Tensor({-10, 20, 20})
end

function OR.train()
	local input, target
	local maxIter = 100000
	local E = torch.Tensor(maxIter)
	local etha = 1.0
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		})
	target = torch.Tensor({
		{0, 1, 1, 1}
		})
	NN.build({2,1})
	for i=1,maxIter do
		NN.forward(input)
		local e = NN.backward(target, 'MSE')
		E[i] = e
		if e < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
		NN.updateParams(etha)
	end
	-- gnuplot.plot(E)
	-- print(NN.getLayer(1))
end

function OR.forward(x, y)
		-- print( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
end

function XOR.set()
	NN.build({2,2,1})
	NN.getLayer(1)[{{1,2}}] = torch.Tensor({{-10, 20, -20}, {-10, -20, 20}})
	NN.getLayer(2)[1] = torch.Tensor({-10, 20, 20})
end

function XOR.train(x, y)
	local input, target
	local maxIter = 100000
	local E = torch.Tensor(maxIter)
	local etha = 1.0
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		})
	target = torch.Tensor({
		{0, 1, 1, 0}
		})
	NN.build({2,2,1})
	for i=1,maxIter do
		NN.forward(input)
		local e = NN.backward(target, 'MSE')
		E[i] = e
		if e < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
		NN.updateParams(etha)
	end
	-- gnuplot.plot(E)
	-- print(NN.getLayer(1))
end

function XOR.forward(x, y)
	-- print( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
	return I2B( NN.forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
end

local function debug()
	NOT.train()
	print("-----------Testing NOT-----------")
	print( (NOT.forward(true) == false) and "Passed" or "Failed")
	print( (NOT.forward(false) == true) and "Passed" or "Failed")

	AND.train()
	print("-----------Testing AND-----------")
	print( (AND.forward(false,false) == false) and "Passed" or "Failed")
	print( (AND.forward(false,true) == false) and "Passed" or "Failed")
	print( (AND.forward(true,false) == false) and "Passed" or "Failed")
	print( (AND.forward(true,true) == true) and "Passed" or "Failed")

	OR.train()
	print("-----------Testing OR-----------")
	print( (OR.forward(false,false) == false) and "Passed" or "Failed")
	print( (OR.forward(false,true) == true) and "Passed" or "Failed")
	print( (OR.forward(true,false) == true) and "Passed" or "Failed")
	print( (OR.forward(true,true) == true) and "Passed" or "Failed")

	XOR.train()
	print("-----------Testing XOR-----------")
	print( (XOR.forward(false,false) == false) and "Passed" or "Failed")
	print( (XOR.forward(false,true) == true) and "Passed" or "Failed")
	print( (XOR.forward(true,false) == true) and "Passed" or "Failed")
	print( (XOR.forward(true,true) == false) and "Passed" or "Failed")
end

debug()

return logicGates