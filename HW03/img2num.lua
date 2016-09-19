require 'math'
require 'torch'

local NN = require 'NeuralNetwork'
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local img2num = {}

local function img2num.train()
	-- body
end

local function img2num.forward(img)
	-- body
end

print(trainset.size, testset.size)

return img2num