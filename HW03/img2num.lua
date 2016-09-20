require 'math'
require 'torch'

local NN = require 'NeuralNetwork'
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local img2num = {}
local oneHotLabels = {}
local trainImgs
local testImgs
local trainLabels
local testLabels

-- Return the oneHot labeled tensor
local function oneHot(mnist_label)
	local one_hot = torch.zeros(10, mnist_label:size(1))
	-- Fix the labels to be 1-10 for indexing in scatter
	mnist_label = (mnist_label+1):view(1,-1):long()
	one_hot:scatter(1, mnist_label, 1)
	return one_hot
end

-- OneHot encode the labels
local function preprocess()
	-- Check if previous encoding already exists?
	local fname = 'labels.asc'
	local f = torch.DiskFile(fname, 'r', true)
	if f == nil then
		-- Do oneHot here and store to the file
		print("Writing label object to disk...")
		trainLabels = oneHot(trainset.label)
		testLabels = oneHot(testset.label)
		table.insert(oneHotLabels, trainLabels)
		table.insert(oneHotLabels, testLabels)
		f = torch.DiskFile(fname, 'w')
		f:writeObject(oneHotLabels)
	else
		print("Reading label object from disk...")
		oneHotLabels = f:readObject()
	end
	trainImgs = trainset.data
	testImgs = testset.data
	print("Preprocessing finished...")
end

function img2num.train()
	preprocess()
	-- 28x28 pixels for each image, 30 hidden neurons, onehot labels
	NN.build({784, 30, 10})
	local maxIter = 100000
	local batchSize = 10
	local maxEpoch = 10
	local eta = 0.01
	local E = torch.Tensor(maxIter)
	-- X is batch input, Y is batch target
	local X = torch.Tensor(784, batchSize)
	local Y = torch.Tensor(10, batchSize)
	for k = 1, maxEpoch do
		-- Per epoch
		-- Shuffle the training set
		local shuffle = torch.randperm(trainset.size)
		for i = 1, trainset.size, batchSize do
			-- Per batch
			for j = 1, batchSize do
				X:select(2,j) = trainImgs[shuffle[ (i-1)*batchSize+j ]]:view(-1,1)
				Y:select(2,j) = trainLabels[shuffle[ (i-1)*batchSize+j ]]
			end
			NN.forward(X)
			NN.backward(Y, 'MSE')
			NN.updateParams(eta)
		end
		-- Check the results
		test()
	end
	gnuplot.plot(E)
end

function img2num.forward(img)
	return NN.forward(img:view(-1,1))
end

local function test()
	local nCorrect = 0
	for i = 1, test.size do
		local l = torch.max(img2num.forward(testImgs[i]),1)[1]
		if l == testset.labels[i] then
			nCorrect = nCorrect+1
		end
	end
	print(nCorrect, "/", test.size)
end

local function debug( )
	img2num.train()
end

debug()

-- print (oneHot(testset.label))
-- print(trainset.size, testset.size)

-- ex = trainset[10]
-- print (ex.x) -- the input (a 28x28 ByteTensor)
-- print (ex.y) -- the label (0--9)

return img2num