require 'math'
require 'torch'
require 'nn'
require 'optim'

-- Network parameters
local batchSize = 10
local maxEpoch = 500
local eta = 0.1
local stopErr = 0.05
local useGPU = false
local cifar10 = false
-- local coarse = true

local cifarTrain
local cifarTest
local nClass

if cifar10 then
	cifarTrain = torch.load('cifar10-train.t7')
	cifarTest = torch.load('cifar10-test.t7')
	nClass = 10
	print('Training on cifar10...')
else
	cifarTrain = torch.load('cifar100-train.t7')
	cifarTest = torch.load('cifar100-test.t7')
	nClass = 100
	print('Training on cifar100...')
end

local img2obj = {}
local oneHotLabels = {}	
local trainImgs = cifarTrain.data
local testImgs = cifarTest.data
local trainLabels
local testLabels
local nTrain
local nTest
local nPlane
local net

-- Return the oneHot labeled tensor
local function oneHot(label)
	-- Fix the labels to be 1-10 for indexing in scatter
	label = (label+1):view(-1,1):long()
	return torch.zeros(label:size(1), nClass):scatter(2, label, 1)
end

-- OneHot encode the labels
local function preprocess()
	nTrain = cifarTrain.label:size()[1]
	nTest = cifarTest.label:size()[1]
	nPlane = cifarTrain.data:size()[2]
	table.insert(oneHotLabels, oneHot(cifarTrain.label))
	table.insert(oneHotLabels, oneHot(cifarTest.label))
	trainLabels = oneHotLabels[1]
	testLabels = oneHotLabels[2]
	trainImgs = trainImgs:float() / 255.0
	print("Preprocessing finished...")
end

local function test()
	local nCorrect = 0
	for i = 1, nTest do
		local l = img2obj.forward(testImgs[i])
		if l-1 == cifarTest.label[i] then
			nCorrect = nCorrect+1
		end
	end
	print(nCorrect, "/", nTest)
	return 1 - (nCorrect / nTest)
end

local function saveNN()
	local fname = 'trainedNetwork.asc'
	local f = torch.DiskFile(fname, 'w')
	f:writeObject(net)
	f:close()
end

local function loadNN()
	local fname = 'trainedNetwork.asc'
	local f = torch.DiskFile(fname, 'r')
	net = f:readObject()
	f:close()	
end

local function train()
	-- X is batch input, Y is batch target
	local X = torch.Tensor(batchSize, nPlane, 32, 32)
	local Y = torch.Tensor(batchSize, nClass)
	local loss = nn.MSECriterion()
	-- Construct LeNet5
	net = nn.Sequential()
	net:add(nn.SpatialConvolution(3,6,5,5))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.Tanh())
	net:add(nn.SpatialConvolution(6,16,5,5))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.Tanh())
	net:add(nn.View(16*5*5))
	net:add(nn.Linear(16*5*5, 120))
	net:add(nn.Tanh())
	net:add(nn.Linear(120,84))
	net:add(nn.Tanh())
	net:add(nn.Linear(84, nClass))

	-- optim stuff
	local theta, gradTheta = net:getParameters()
	local optimState = {learningRate = eta}

	print("Start training with CPU...")
	for k = 1, maxEpoch do
		-- Per epoch
		print("Epoch", k)

		local shuffle = torch.randperm(nTrain)
		for i = 1, nTrain /batchSize do
			for j=1,batchSize do
				X[j] = trainImgs[shuffle[ i ]]
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

		-- Check the results
		if test() < stopErr then
			print("Training finished at epoch", k)
			break
		end
		saveNN()
	end
end

function img2obj.train()
	preprocess()
	train()
end

function img2obj.forward(img)
	img = img:float() / 255.0
	local X = torch.Tensor(1,nPlane,32,32)
	X[1] = img
	local _,l = torch.max(net:forward(X), 1)
	return math.floor(l[1])
end

function img2obj.view(img)
	l = img2obj.forward(img)
	require 'camera'
	require 'image'
	toDisp = img:clone()
	toDisp = image.scale(toDisp, 640, 640) 
	toDisp = image.drawText(toDisp, tostring(l-1), 30, 30, {color = {255, 0, 0}, size = 5})
	image.display(toDisp)
end

function img2obj.cam(idx)
	-- Default idx to 0
	local idx = idx == nil and 0 or idx
	require 'camera'
	require 'image'
	local cam = image.Camera{}
	while true do
		local frame = cam:forward()
		frame = frame * 255.0
		img2obj.view(image.scale(frame, 32, 32))
	end
	cam:stop()
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

return img2obj
