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
	trainImgs = trainImgs:float() / 255.0
	print("Preprocessing finished...")
end

local function test()
	local nCorrect = 0
	for i = 1, testset.size do
		local l = img2num.forward(testImgs[i])
		if l-1 == testset.label[i] then
			nCorrect = nCorrect+1
		end
	end
	print(nCorrect, "/", testset.size)
	return 1 - (nCorrect / testset.size)
end

local function train()
	-- X is batch input, Y is batch target
	local X = torch.Tensor(batchSize, 1, 28, 28)
	local Y = torch.Tensor(batchSize, 10)
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
	local optimState = {learningRate = 0.01}

	-- if useGPU then
	-- 	trainImgs:cuda()

	print("Start training with CPU...")
	for k = 1, maxEpoch do
		-- Per epoch
		print("Epoch", k)

		local shuffle = torch.randperm(trainset.size)
		for i = 1, trainset.size/batchSize do
			for j=1,batchSize do
				X[{j,1}] = trainImgs[shuffle[ i ]]
				Y[j] = trainLabels[shuffle[ i ]]:view(1,-1)
			end
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

		-- local timer = torch.Timer()
		-- Check the results
		if test() < stopErr then
			print("Training finished at epoch", k)
			break
		end
		-- print("Inference time: ", timer:time().real)
	end
end

function img2num.train()
	preprocess()
	train()
end

function img2num.forward(img)
	img = img:float() / 255.0
	local X = torch.Tensor(1,1,28,28)
	X[{1,1}] = img
	local _,l = torch.max(net:forward(X), 1)
	return math.floor(l[1])
end

local function benchmark()
	-- Average time for one epoch
	preprocess()
	local timer = torch.Timer()
	train()
	local time = timer:time().real
	print('For each epoch, CPU took ', time/maxEpoch, 's.')
	local timer = torch.Timer()
end

-- benchmark()

return img2num
