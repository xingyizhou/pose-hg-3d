--
--  Original version: Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--  (Modified a bit by Alejandro Newell)
--  (Modified by Xingyi Zhou)
-- 
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt, dataset, ref)
   -- The train and valid loader
   local loaders = {}
   for i, split in ipairs{'train', 'valid'} do
         loaders[split] = M.DataLoader(opt, dataset, ref, split)
   end
   return loaders
end

function DataLoader:__init(opt, dataset, ref, split)
    local function preinit()
        paths.dofile('dataset/' .. opt.dataset .. '.lua')
    end

    local function init()
        _G.opt, _G.dataset, _G.ref, _G.split = opt, dataset, ref, split
        paths.dofile('../ref.lua')
    end

    local function main(idx)
        torch.setnumthreads(1)
        return dataset:size(split)
    end

    local threads, sizes = Threads(opt.nThreads, preinit, init, main)
    self.threads = threads
    self.batchsize = opt[split .. 'Batch']
    self.split = split
    
    self.nsamples = {valid = {h36m = opt.validIters['h36m'], mpii = opt.validIters['mpii'], mpi_inf_3dhp = opt.validIters['mpi_inf_3dhp'], demo = 1}, train = sizes[1][1]}
    self.iters = {h36m = opt.validIters['h36m'], mpii = opt.validIters['mpii'], train = opt[split .. 'Iters'], mpi_inf_3dhp = opt.validIters['mpi_inf_3dhp'], demo = 1}
    
end

function DataLoader:size()
    return self.iters
end

function DataLoader:run(tmp)
    local threads = self.threads
    local size = self.iters[tmp] * self.batchsize

    local idxs = {}
    idxs['h36m'] = torch.range(1,self.nsamples[self.split]['h36m'])
    idxs['mpii'] = torch.range(1,self.nsamples[self.split]['mpii'])
    idxs['mpi_inf_3dhp'] = torch.range(1,self.nsamples[self.split]['mpi_inf_3dhp'])

    for i = 2,math.ceil(size/self.nsamples[self.split]['h36m']) do
        idxs['h36m'] = idxs['h36m']:cat(torch.range(1,self.nsamples[self.split]['h36m']))
    end
    for i = 2,math.ceil(size/self.nsamples[self.split]['mpii']) do
        idxs['mpii'] = idxs['mpii']:cat(torch.range(1,self.nsamples[self.split]['mpii']))
    end

    -- Shuffle indices
    if self.split == 'train' then
        idxs['h36m'] = idxs['h36m']:index(1,torch.randperm(idxs['h36m']:size(1)):long())
        idxs['mpii'] = idxs['mpii']:index(1,torch.randperm(idxs['mpii']:size(1)):long()) 
    end
    -- Map indices to training/validation/test split
    idxs['h36m'] = opt.idxRef['h36m'][self.split]:index(1,idxs['h36m']:long())
    idxs['mpii'] = opt.idxRef['mpii'][self.split]:index(1,idxs['mpii']:long())
    idxs['mpi_inf_3dhp'] = opt.idxRef['mpi_inf_3dhp'][self.split]:index(1,idxs['mpi_inf_3dhp']:long())
    idxs['demo'] = torch.range(1, 1)
    
    local n, idx, sample = 0, 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local len = math.min(self.batchsize, size - idx + 1)
            local indices = torch.zeros(len)
            local tps = {}
            for i = 1, len do 
                local rd = (torch.random() % 10 < opt.Ratio3D and 'h36m') or 'mpii'
                local tp = self.tpValid or rd
                indices[i] = idxs[tp][idx]
                tps[i] = tp
                idx = idx + 1
            end
            threads:addjob(
                function(indices)
                    local inp,out = _G.loadData(_G.split, indices, tps)
                    collectgarbage()
                    return {inp,out,indices}
                end,
                function(_sample_) sample = _sample_ end, indices
            )
        end
    end

    local function loop()
        enqueue()
        if not threads:hasjob() then return nil end
        threads:dojob()
        if threads:haserror() then threads:synchronize() end
        enqueue()
        n = n + 1
        return n, sample
    end
    return loop
end

return M.DataLoader
