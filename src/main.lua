require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('model.lua')   -- Read in network model
paths.dofile('util/pyTools.lua')
torch.setnumthreads(1)
local Dataloder
paths.dofile('train.lua')
Dataloader = paths.dofile('util/multi-dataloader.lua')
loader = Dataloader.create(opt, dataset, ref)

-- Initialize logs
ref.log = {}
ref.log.train = Logger(paths.concat(opt.save, 'train.log'), opt.continue)
ref.log.valid = Logger(paths.concat(opt.save, 'valid.log'), opt.continue)

-- Main training loop
if not (opt.demo == '') then
    demo()
else
    for i=1,opt.nEpochs do
        print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
        train()
        if epoch % opt.validEpoch == 0 then
            valid()
        end
        epoch = epoch + 1
        collectgarbage()
    end
end

-- Update reference for last epoch
opt.lastEpoch = epoch - 1

-- Save model
model:clearState()
torch.save(paths.concat(opt.save,'options.t7'), opt)
torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
torch.save(paths.concat(opt.save,'final_model.t7'), model)

-- Generate final predictions on validation set
if opt.finalPredictions then
	ref.log = {}
	loader.test = Dataloader(opt, dataset, ref, 'test')
	predict()
end
