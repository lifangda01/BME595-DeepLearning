--[[BME 595: Deep Learning
	Final Project
	Author: Axel Masquelin, Fangda Li
	Description: Training Script for AlexNet, GoogleNet, and CNN Network.
	]]
require 'optim'
require 'math'
require 'torch'
require 'nn'
require 'image'   -- to visualize the dataset
require 'cutorch'
require 'cudnn'
require 'inn'
require 'gnuplot'
local dl = require 'dataload'

dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train or test on breast metastasis dataset.')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'train a network')
cmd:option('-test', false, 'test a network')
cmd:option('-save', '', 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'resnet18', 'type of model: resnet18 | resnet101')
cmd:option('-full', false, 'use full dataset')
cmd:option('-preprocess', false, 'update the mean and std of dataset')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-learningRateDecay', 5e-7, 'learning rate decay')
cmd:option('-batchSize', 2, 'mini-batch size (1 = pure stochastic)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-maxEpoch', 5, 'maximum number of epochs')
cmd:option('-deadState', 1.0, 'acceptable loss to stop training')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)


local nClass = 2
local nPixel = 224
local dlTrain, dlTest
if opt.full then
	dlTrain = dl.ImageClass("./dataset-full/train", {3, nPixel, nPixel})
	dlTest = dl.ImageClass("./dataset-full/test", {3, nPixel, nPixel})
else
	dlTrain = dl.ImageClass("./dataset-small/train", {3, nPixel, nPixel})
	dlTest = dl.ImageClass("./dataset-small/test", {3, nPixel, nPixel})
end
local nTrain = dlTrain:size()
local nTest = dlTest:size()
local net

local function savenet(fname)
	local f = torch.DiskFile(fname, 'w')
	f:writeObject(net)
	f:close()
end

local function loadnet(fname)
	local f = torch.DiskFile(fname, 'r')
	net = f:readObject()
	f:close()	
end

if opt.network == '' then
	if opt.model == 'resnet18' then
		net = torch.load'resnet-18.t7'
		net:remove(11)
		net:add(nn.Linear(512,nClass)):add(nn.LogSoftMax()):cuda()
	elseif opt.model == 'resnet101' then
		net = torch.load'resnet-101.t7'
		net:remove(11)
		net:add(nn.Linear(2048,nClass)):add(nn.LogSoftMax()):cuda()
	else
		print('Model not recognized.')
		cmd:text()
		error()
	end
else
	loadnet(opt.network)
end
print(net)
-- Training parameters
local batchSize = opt.batchSize
local maxEpoch = opt.maxEpoch
local deadState = opt.deadState  --Loop termination case, where lessor rate is small enough.
local optimState = {learningRate = opt.learningRate,
                    momentum = opt.momentum,
                    learningRateDecay = opt.learningRateDecay}
-- Normalization parameters, obtained from preprocess()
local mean_rgb = {0.66094456716265, 0.48451634488599, 0.73816294143024}
local std_rgb = {0.12633086558123, 0.14390315048536, 0.08288355023738}


local function test(threshold)
	local nCorrect = 0
	local nTP = 0
	local nFP = 0
	local label = 0
	local nNormalTest = dlTest.classList[1]:size()[1]
	local nTumorTest = dlTest.classList[2]:size()[1]
	local confusion = optim.ConfusionMatrix({'normal', 'tumor'})
	print("Testing started...")
	for k, X, y in dlTest:subiter(batchSize,nTest) do
		xlua.progress(k, nTest)
		X = X:cuda()

		for j = 1,3 do
			X[{ {},j,{},{} }]:add(-mean_rgb[j])
			X[{ {},j,{},{} }]:div(std_rgb[j])
		end
		-- Normalization within batch
		y = y:cuda()
		h = net:forward(X)
		h:exp()
		for i = 1,h:size()[1] do
			if h[{i,2}] >= threshold then
				confusion:add(2,y[i])
				if y[i] == 2 then
					nTP = nTP + 1
					nCorrect = nCorrect+1
				else
					nFP = nFP + 1					
				end
			elseif y[i] == 1 then
				confusion:add(1,y[i])
				nCorrect = nCorrect+1
			end
		end
	end
	-- Accuracy, False Positive, True Positive, Precision, Recall
	local accu, fp, tp, pr = nCorrect / nTest, nFP / nNormalTest, nTP / nTumorTest, nTP / (nTP+nFP)
	print(string.format("thresh = %.2f, accu = %.4f, fp = %.4f, tp = %.4f, pr = %.4f", threshold, accu, fp, tp, pr))
	print(confusion)
	return accu, fp, tp, pr
end

local function preprocess()
	print('Normalizing dataset...')
	local n = 0
	local sqsum_rgb = {0, 0, 0}
	for k, X, y in dlTrain:subiter() do
		xlua.progress(k, nTrain)
		n = n + 1
		for i=1,3 do
			local mean = X[{ {},i,{},{} }]:mean()
			local delta = mean - mean_rgb[i]
			mean_rgb[i] = mean_rgb[i] + delta / n
			-- local delta2 = mean - mean_rgb[i]
			-- M2_rgb[i] = M2_rgb[i] + delta*delta2
		end
	end
	for k, X, y in dlTrain:subiter() do
		xlua.progress(k, nTrain)
		n = n + 1
		for i=1,3 do
			sqsum_rgb[i] = sqsum_rgb[i] + X[{ {},i,{},{} }]:add(-mean_rgb[i]):pow(2):sum()
		end
	end
	for i=1,3 do
		std_rgb[i] = math.sqrt(sqsum_rgb[i] / nTrain / (3*nPixel*nPixel))
	end
end

-- Training Function --
local function train()
	if opt.preprocess then
		preprocess()
	end
	local criterion = nn.ClassNLLCriterion():cuda()
	local theta, gradTheta = net:getParameters()
	local batchNum = 0
	local epoch = 0
	local totalLoss = 99.0
	print('Training started...')
	while totalLoss >= deadState and epoch < maxEpoch do
		totalLoss = 0
		local shuffle = torch.randperm(nTrain)
		for i = 1, math.floor(nTrain / batchSize) do
			xlua.progress(i, math.floor(nTrain / batchSize))
			indices = shuffle[{{(i-1)*batchSize+1, batchSize*i}}]
			X, Y = dlTrain:index(indices)
			X = X:cuda()
			-- Global normalization
			for j = 1,3 do
				X[{ {},j,{},{} }]:add(-mean_rgb[j])
				X[{ {},j,{},{} }]:div(std_rgb[j])
			end			
			Y = Y:cuda()
			-- NLL stuff
			feval = function (theta)
				gradTheta:zero()
				local f = 0
				-- NLL doesn't accommodate batch loss
				for i = 1,batchSize do
					local output = net:forward(X[i]:view(1,3,224,224))
					local loss = criterion:forward(output, Y[i])
					f = f + loss
					local gradLoss = criterion:backward(output, Y[i])
					net:backward(X[i]:view(1,3,224,224), gradLoss)
					totalLoss = totalLoss + loss
				end
				f = f / batchSize
				gradTheta:div(batchSize)
				return f, gradTheta
			end

			optim.sgd(feval, theta, optimState)
			batchNum = batchNum + 1 --Updating Batch Counter
		end
		epoch = epoch + 1 --Updating Epoch Counter
		test(0.7)
		print(string.format('\nEpoch %d, loss = %f', epoch, totalLoss))
		savenet(string.format('%s.net',opt.model))
	end
end

local function generateROC()
	local threshold = 0.0
	local step = 0.02
	local numSteps = 1/step + 1
	local maxAccu = 0
	local FPRs = torch.Tensor(numSteps)
	local TPRs = torch.Tensor(numSteps)
	for i = 1,numSteps do 
		accu, fp, tp, pr = test(threshold)
		FPRs[i] = fp
		TPRs[i] = tp
		if accu > maxAccu then 
			maxAccu = accu 
			print(string.format('New max accuracy %.4f achieved at threshold %.2f', accu, threshold))
		end
		threshold = threshold + step
	end
	torch.save(string.format('TPR_%s.t7', opt.model), TPRs)
	torch.save(string.format('FPR_%s.t7', opt.model), FPRs)
	print('Final max accuracy '..maxAccu)
	gnuplot.svgfigure(string.format('ROC_%s.svg', opt.model))
	gnuplot.plot('ResNet18', FPRs, TPRs, '+-')
	gnuplot.xlabel('False Positive Rate')
	gnuplot.ylabel('True Positive Rate')
	gnuplot.title('Receiver Operating Characteristics')
	gnuplot.plotflush()
end

if opt.train then
	train()
end
if opt.test then
	generateROC()
end