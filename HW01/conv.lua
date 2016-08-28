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

ffi = require('ffi')
C = ffi.load(paths.cwd() .. '/libconv.so')
ffi.cdef [[
	void conv(double *x, double *k, double *r, int x_c, int r_r, int r_c, int k_s)
]]

function conv.C_conv(x, k)
	r = torch.DoubleTensor(x:size(1)-k:size(1)+1, x:size(2)-k:size(2)+1)
	C.conv(torch.data(x), torch.data(k), torch.data(r), x:size(2), x:size(1)-k:size(1)+1, x:size(2)-k:size(2)+1, k:size(1))
	return r
end

return conv