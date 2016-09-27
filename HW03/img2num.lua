require 'math'
require 'torch'

local NN = require 'NeuralNetwork'
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local img2num = {}
local oneHotLabels = {}	
local trainImgs = trainset.data
local testImgs = testset.data
local trainLabels
local testLabels

-- Return the oneHot labeled tensor
local function oneHot(mnist_label)
	-- Fix the labels to be 1-10 for indexing in scatter
	mnist_label = (mnist_label+1):view(-1,1):long()
	return torch.zeros(mnist_label:size(1), 10):scatter(2, mnist_label, 1)
end

-- OneHot encode the labels
local function preprocess()
	-- Check if previous encoding already exists?
	local fname = 'labels.asc'
	local f = torch.DiskFile(fname, 'r', true)
	if f == nil then
		-- Do oneHot here and store to the file
		print("Writing label object to disk...")
		table.insert(oneHotLabels, oneHot(trainset.label))
		table.insert(oneHotLabels, oneHot(testset.label))
		f = torch.DiskFile(fname, 'w')
		f:writeObject(oneHotLabels)
		f:close()
	else
		print("Reading label object from disk...")
		oneHotLabels = f:readObject()
	end
	trainLabels = oneHotLabels[1]
	testLabels = oneHotLabels[2]
	print("Preprocessing finished...")
end

local function test()
	local nCorrect = 0
	for i = 1, testset.size do
		-- Find the max along the row axis
		local _, l = torch.max(img2num.forward(testImgs[i]:double()),1)
		if l[1][1]-1 == testset.label[i] then
			nCorrect = nCorrect+1
		end
	end
	print(nCorrect, "/", testset.size)
end

local function saveNN()
	local fname = 'trainedNetwork.asc'
	local f = torch.DiskFile(fname, 'w')
	f:writeObject(NN)
	f:close()
end

local function loadNN()
	local fname = 'trainedNetwork.asc'
	local f = torch.DiskFile(fname, 'r')
	NN = f:readObject()
	f:close()	
end

function img2num.train()
	preprocess()
	-- 28x28 pixels for each image, 30 hidden neurons, onehot labels
	NN.build({784, 30, 10})
	local batchSize = 10
	local maxEpoch = 30
	local eta = 0.05
	-- X is batch input, Y is batch target
	local X = torch.Tensor(batchSize, 784)
	local Y = torch.Tensor(batchSize, 10)
	for k = 1, maxEpoch do
		-- Per epoch
		print("Epoch", k)
		-- Shuffle the training set
		local shuffle = torch.randperm(trainset.size)
		for i = 1, trainset.size/batchSize do
			-- Per batch
			for j = 1, batchSize do
				X[j] = trainImgs[shuffle[ (i-1)*batchSize+j ]]:view(1,-1):double() / 255.0
				Y[j] = trainLabels[shuffle[ (i-1)*batchSize+j ]]:view(1,-1):double()
			end
			NN.forward(X:t())
			NN.backward(Y:t(), 'MSE')
			NN.updateParams(eta)
		end
		-- Check the results
		test()
	end
end

function img2num.forward(img)
	return NN.forward(img:view(-1,1))
end

local function debug()
	timer = torch.Timer()
	img2num.train()
	print('Training took ', timer:time().real, 's.')
end

debug()

return img2num