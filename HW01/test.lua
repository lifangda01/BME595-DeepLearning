-- Test bench for convolution in Lua

require 'torch'
require 'image'
conv = require('conv')

local epsilon = 1e-9
local num_tests = 5

-- To save an image:
-- res = res:add(-res:min()):div(res:max()-res:min())
-- image.save('result.png',res)

local function C_test(x, k, ans)
	timer = torch.Timer()
	res = conv.C_conv(x, k)
	time = timer:time().real
	print(math.abs(torch.sum(res - ans)) < epsilon and 'Passed' or 'Failed')
	return time
end

local function Lua_test(x, k, ans)
	timer = torch.Timer()
	res = conv.Lua_conv(x, k)
	time = timer:time().real
	print(math.abs(torch.sum(res - ans)) < epsilon and 'Passed' or 'Failed')
	return time
end

local function test()
	Lua_duration = torch.DoubleTensor(3,num_tests)
	C_duration = torch.DoubleTensor(3,num_tests)
	for i=1,num_tests do
		x1 = torch.rand(10,10)
		x1 = image.load('images/lena_128.jpg', 1, 'double'):select(1,1)
		k1 = torch.rand(9,9)
		ans1 = torch.conv2(x1,k1)
		C_duration[1][i] = C_test(x1,k1,ans1)
		Lua_duration[1][i] = Lua_test(x1,k1,ans1)
		-- x2 = torch.rand(100,100)
		x2 = image.load('images/lena_256.jpg', 1, 'double'):select(1,1)
		k2 = torch.rand(9,9)
		ans2 = torch.conv2(x2,k2)
		C_duration[2][i] = C_test(x2,k2,ans2)
		Lua_duration[2][i] = Lua_test(x2,k2,ans2)

		x3 = torch.rand(1000,1000)
		x3 = image.load('images/lena_512.jpg', 1, 'double'):select(1,1)
		k3 = torch.rand(9,9)
		ans3 = torch.conv2(x3,k3)
		C_duration[3][i] = C_test(x3,k3,ans3)
		Lua_duration[3][i] = Lua_test(x3,k3,ans3)
	end
	C_avg = torch.sum(C_duration,2) / num_tests
	Lua_avg = torch.sum(Lua_duration,2) / num_tests
	print(C_avg)
	print(Lua_avg)
end

test()


