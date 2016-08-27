-- Test bench for convolution in Lua

require 'conv'
require 'torch'

local epsilon = 1e-12

print("Test 1:", torch.sum(conv.Lua_conv(t, k) - torch.conv2(t,k)) < epsilon and 'Passed' or 'Failed')
