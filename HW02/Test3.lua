local l = require 'logicGates'

local comb = {
   {false, false},
   {false, true},
   {true, false},
   {true, true},
}

for _, c in ipairs(comb) do
   assert(l.AND(unpack(c)) == (c[1] and c[2]), 'Bad AND')
end
for _, c in ipairs(comb) do
   assert(l.OR(unpack(c)) == (c[1] or c[2]), 'Bad OR')
end
for _, c in ipairs(comb) do
   assert(l.NOT(c[1]) == not c[1], 'Bad NOT')
end
local function XOR(a, b) return (a and not b) or (not a and b) end
for _, c in ipairs(comb) do
   assert(l.XOR(unpack(c)) == XOR(unpack(c)), 'Bad XOR')
end