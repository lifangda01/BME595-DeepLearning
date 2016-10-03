require 'math'
require 'torch'
require 'nn'

local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local img2num = {}
local oneHotLabels = {}	
local trainImgs = trainset.data
local testImgs = testset.data
local trainLabels
local testLabels
local net

-- Network parameters
local batchSize = 10
local maxEpoch = 30
local eta = 0.05
local stopErr = 0.05
local useGPU = false

-- Return the oneHot labeled tensor
local function oneHot(mnist_label)
	-- Fix the labels to be 1-10 for indexing in scatter
	mnist_label = (mnist_label+1):view(-1,1):long()
	return torch.zeros(mnist_label:size(1), 10):scatter(2, mnist_label, 1)
end

-- OneHot encode the labels
local function preprocess()
	table.insert(oneHotLabels, oneHot(trainset.label))
	table.insert(oneHotLabels, oneHot(testset.label))
	trainLabels = oneHotLabels[1]
	testLabels = oneHotLabels[2]
	print("Preprocessing finished...")
end

local function testWithCPU()
	local nCorrect = 0
	local X = torch.Tensor(1,28,28)
	for i = 1, testset.size do
		-- Find the max along the column axis
		X[1] = testImgs[i]
		local _,l = torch.max(img2num.forward(X / 255.0), 1)
		if l[1]-1 == testset.label[i] then
			nCorrect = nCorrect+1
		end
	end
	print(nCorrect, "/", testset.size)
	return 1 - (nCorrect / testset.size)
end

local function trainWithCPU()
	-- X is batch input, Y is batch target
	local X = torch.Tensor(1, 28, 28)
	local Y = torch.Tensor(10)
	local loss = nn.MSECriterion()
	-- Construct LeNet5
	net = nn.Sequential()
	net:add(nn.SpatialConvolution(1,6,5,5))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.Tanh())
	net:add(nn.SpatialConvolution(6,16,5,5))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.Tanh())
	net:add(nn.View(16*4*4))
	net:add(nn.Linear(16*4*4, 120))
	net:add(nn.Tanh())
	net:add(nn.Linear(120,84))
	net:add(nn.Tanh())
	net:add(nn.Linear(84, 10))

	-- optim stuff
	local optim = require 'optim'
	local theta, gradTheta = net:getParameters()
	local optimState = {learningRate = 0.1}

	print("Start training with CPU...")
	for k = 1, maxEpoch do
		-- Per epoch
		print("Epoch", k)

		-- Shuffle the training set
		-- local shuffle = torch.randperm(trainset.size)
		-- for i = 1, trainset.size do
		-- 	-- Per batch
		-- 	net:zeroGradParameters()
			-- X[1] = trainImgs[shuffle[ i ]] / 255.0
			-- Y = trainLabels[shuffle[ i ]]:view(1,-1)
		-- 	local pred = net:forward(X)
		-- 	local err = loss:forward(pred, Y)
		-- 	local grad = loss:backward(pred, Y)
		-- 	net:backward(X, grad)
		-- 	net:updateParameters(eta)
		-- end

		local shuffle = torch.randperm(trainset.size)
		for i = 1, trainset.size do
			X[1] = trainImgs[shuffle[ i ]] / 255.0
			Y = trainLabels[shuffle[ i ]]:view(1,-1)
			local function feval(theta)
				gradTheta:zero()
				local hx = net:forward(X)
				local J = loss:forward(hx, Y)
				local dJ_dhx = loss:backward(hx, Y)
				net:backward(X, dJ_dhx)
				return J, gradTheta
			end
			optim.sgd(feval, theta, optimState)
		end

		-- Check the results
		if testWithCPU() < stopErr then
			print("Training finished at epoch", k)
			break
		end
	end
end

function img2num.train()
	preprocess()
	trainWithCPU()
end

function img2num.forward(img)
	return net:forward(img)
end

local function benchmark()
	-- Average time for one epoch
	preprocess()
	local timer = torch.Timer()
	trainWithCPU()
	local cpuTime = timer:time().real
	print('For each epoch, CPU took ', cpuTime/maxEpoch, 's.')
	local timer = torch.Timer()
end

benchmark()

return img2num
