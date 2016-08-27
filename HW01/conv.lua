-- Convolution in Lua

require 'math'
require 'torch'

local conv = {}

local function pixel_conv(x, k, i, j)
	k_s = k:size(1)
	h_s = math.floor(k_s/2)
	result = 0
	for m = 0, k_s-1 do
		for n = 0, k_s-1 do
			result = result + x[i-h_s+m][j-h_s+n] * k[k_s-m][k_s-n]
		end
	end
	return result
end

function conv.Lua_conv(x, k)
	h_s = math.floor(k:size(1)/2)
	r = torch.DoubleTensor(x:size(1)-k:size(1)+1, x:size(2)-k:size(2)+1)
	for i = 1, r:size(1) do
		for j = 1, r:size(2) do
			r[i][j] = pixel_conv(x, k, i+h_s, j+h_s)
		end
	end
	return r
end

local epsilon = 1e-12

local x = torch.rand(10,10)
local k = torch.rand(9,9)
print("Test 1:", torch.sum(conv.Lua_conv(x, k) - torch.conv2(x,k)) < epsilon and 'Passed' or 'Failed')
local x = torch.rand(100,100)
local k = torch.rand(9,9)
print("Test 2:", torch.sum(conv.Lua_conv(x, k) - torch.conv2(x,k)) < epsilon and 'Passed' or 'Failed')
local x = torch.rand(1000,1000)
local k = torch.rand(9,9)
print("Test 3:", torch.sum(conv.Lua_conv(x, k) - torch.conv2(x,k)) < epsilon and 'Passed' or 'Failed')

return conv