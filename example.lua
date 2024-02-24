require('torch')
require('nn')
require 'torch'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
require 'stn'
local sanitize = require('sanitize')
network = torch.load('Trained_Network_2.bin')
torch.save('Trained_Network_2.bin', sanitize(network))

