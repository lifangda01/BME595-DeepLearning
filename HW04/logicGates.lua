require 'math'
require 'nn'
require 'torch'

local NOT = {}
local AND = {}
local OR = {}
local XOR = {}
local logicGates = {NOT, AND, OR, XOR}
local epsilon = 1e-4
local not_net, and_net, or_net, xor_net

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

function NOT.train()
	local input, target
	local maxIter = 100000
	local eta = 100.0
	input = torch.Tensor({
		{0, 1}
		}):t()
	target = torch.Tensor({
		{1, 0}
		}):t()
	not_net = nn.Sequential()
	local loss = nn.MSECriterion()
	local sig = nn.Sigmoid()
	local lin = nn.Linear(1,1)
	not_net:add(lin)
	not_net:add(sig)
	for i=1,maxIter do
		local pred = not_net:forward(input)
		local err = loss:forward(pred, target)
		local grad = loss:backward(pred, target)
		not_net:zeroGradParameters()
		not_net:backward(input, grad)
		not_net:updateParameters(eta)
		if err < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
	end
end

function NOT.forward(x)
	return I2B( not_net:forward(torch.Tensor({ B2I(x) })) : view(-1,1)[1][1] )
end

function AND.train()
	local input, target
	local maxIter = 100000
	local eta = 10.0
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		}):t()
	target = torch.Tensor({
		{0, 0, 0, 1}
		}):t()
	and_net = nn.Sequential()
	local loss = nn.MSECriterion()
	local sig = nn.Sigmoid()
	local lin = nn.Linear(2,1)
	and_net:add(lin)
	and_net:add(sig)
	for i=1,maxIter do
		local pred = and_net:forward(input)
		local err = loss:forward(pred, target)
		local grad = loss:backward(pred, target)
		and_net:zeroGradParameters()
		and_net:backward(input, grad)
		and_net:updateParameters(eta)
		if err < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
	end
end

function AND.forward(x, y)
	return I2B( and_net:forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
end

function OR.train()
	local input, target
	local maxIter = 100000
	local eta = 10.0
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		}):t()
	target = torch.Tensor({
		{0, 1, 1, 1}
		}):t()
	or_net = nn.Sequential()
	local loss = nn.MSECriterion()
	local sig = nn.Sigmoid()
	local lin = nn.Linear(2,1)
	or_net:add(lin)
	or_net:add(sig)
	for i=1,maxIter do
		local pred = or_net:forward(input)
		local err = loss:forward(pred, target)
		local grad = loss:backward(pred, target)
		or_net:zeroGradParameters()
		or_net:backward(input, grad)
		or_net:updateParameters(eta)
		if err < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
	end
end

function OR.forward(x, y)
	return I2B( or_net:forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
end

function XOR.train(x, y)
	local input, target
	local maxIter = 100000
	local eta = 0.5
	input = torch.Tensor({
		{0, 0, 1, 1},
		{0, 1, 0, 1}
		}):t()
	target = torch.Tensor({
		{0, 1, 1, 0}
		}):t()
	xor_net = nn.Sequential()
	local loss = nn.MSECriterion()
	local sig = nn.Sigmoid()
	local lin1 = nn.Linear(2,2)
	local lin2 = nn.Linear(2,1)
	xor_net:add(lin1)
	xor_net:add(sig)
	xor_net:add(lin2)
	-- xor_net:add(sig)
	for i=1,maxIter do
		local pred = xor_net:forward(input)
		local err = loss:forward(pred, target)
		local grad = loss:backward(pred, target)
		xor_net:zeroGradParameters()
		xor_net:backward(input, grad)
		xor_net:updateParameters(eta)
		if err < epsilon then 
			print("Training finished at iteration", i)
			break 
		end
	end
end

function XOR.forward(x, y)
	return I2B( xor_net:forward(torch.Tensor({ B2I(x), B2I(y) })) : view(-1,1)[1][1] )
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

-- debug()

return logicGates
