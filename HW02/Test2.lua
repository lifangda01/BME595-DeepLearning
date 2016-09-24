local net = require 'NeuralNetwork'
local L = 3 + math.random(7)
local s = torch.Tensor(L):random(10)
local m = 1 + math.random(9)
net.build(s:totable())

require 'nn'
local myNet = nn.Sequential()
for l = 1, L - 1 do
   myNet:add(nn.Linear(s[l], s[l+1]))
   myNet:add(nn.Sigmoid())
end

for l = 1, L - 1 do
   local k = 1 + 2 * (l - 1)
   myNet:get(k).weight:copy(net.getLayer(l)[{ {}, {2, -1} }])
   myNet:get(k).bias:copy(net.getLayer(l)[{ {}, 1 }])
end

local X = torch.randn(m, s[1])
assert(torch.dist(myNet:forward(X):t(), net.forward(X:t())) < 1e-3, 'Fail')