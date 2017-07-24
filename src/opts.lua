if not opt then

projectDir = projectDir or paths.concat(os.getenv('HOME'),'pose-hg-3d')

local function parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-demo',          '', 'demo path')
    cmd:option('-disable3DHP',       false, 'not validate on h36m')
    cmd:option('-validH36M',       true, 'not validate on h36m')
    cmd:option('-valid3DHP',       true, 'validate on mpi-inf-3dhp')
    cmd:option('-DEBUG',       0, 'debug')
    cmd:option('-display',       10, 'Display Loss')
    cmd:option('-Ratio3D',       5, 'Ratio of 3D data')
    cmd:option('-expID',       'default', 'Experiment ID')
    cmd:option('-dataset',        'fusion', 'Dataset choice: mpii | fusion | h36m')
    cmd:option('-h36mFullTest',        false, 'full test')
    cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
    cmd:option('-mpiiImgDir',  paths.concat(os.getenv('HOME'),'Datasets/mpii/images'))
    cmd:option('-h36mImgDir',  paths.concat(os.getenv('HOME'),'Datasets/Human3.6M/images'))
    cmd:option('-mpi_inf_3dhpImgDir',  paths.concat(os.getenv('HOME'),'Datasets/MPI-INF-3DHP/images'))
    cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('-manualSeed',         -1, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-finalPredictions',false, 'Generate a final set of predictions at the end of training (default no)')
    cmd:option('-nThreads',            4, 'Number of data loading threads')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-gt2D',          false, 'gt 2d for constraint')
    cmd:option('-netType',          'hgreg-3d', 'Options: hg-reg-3d')
    cmd:option('-loadModel',      'none', 'Provide full path to a previously trained model')
    cmd:option('-continue',        false, 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off')
    cmd:option('-task',           'pose-hgreg-3d', 'Network task: pose-3d-reg')
    cmd:option('-nFeats',            256, 'Number of features in the hourglass')
    cmd:option('-nStack',              2, 'Number of hourglasses to stack')
    cmd:option('-nModules',            2, 'Number of residual modules at each location in the hourglass')
    cmd:option('-nRegModules',            2, 'Number of residual modules at each location after the hourglass')
    cmd:text()
    cmd:text(' ---------- Snapshot options -----------------------------------')
    cmd:text()
    cmd:option('-validEpoch',            5, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-snapshot',            5, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-saveInput',       false, 'Save input to the network (useful for debugging)')
    cmd:option('-saveHeatmaps',    false, 'Save output heatmaps')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-varWeight',             0.01, 'Weakly supervised Weight')
    cmd:option('-regWeight',             0.1, 'Regression Weight')
    cmd:option('-PCK_Threshold',             150, 'PCK_Threshold')
    cmd:option('-dropLR',             10000, 'Drop Learning rate')
    cmd:option('-LR',             2.5e-4, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-alpha',            0.99, 'Alpha')
    cmd:option('-epsilon',          1e-8, 'Epsilon')
    cmd:option('-crit',            'MSE', 'Criterion type')
    cmd:option('-optMethod',   'rmsprop', 'Optimization method: rmsprop | sgd | nag | adadelta')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-nEpochs',           60, 'Total number of epochs to run')
    cmd:option('-trainIters',       4000, 'Number of train iterations per epoch')
    cmd:option('-trainBatch',          6, 'Mini-batch size')
    cmd:option('-validIters',       2958, 'Number of validation iterations per epoch')
    cmd:option('-validBatch',          1, 'Mini-batch size for validation')
    cmd:option('-nValidImgs',       2958, 'Number of images to use for validation. Only relevant if randomValid is set to true')
    cmd:option('-randomValid',     false, 'Whether or not to use a fixed validation set of 2958 images (same as Tompson et al. 2015)')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',          256, 'Input image resolution')
    cmd:option('-outputRes',          64, 'Output heatmap resolution')
    cmd:option('-scale',             .25, 'Degree of scale augmentation')
    cmd:option('-rotate',             30, 'Degree of rotation augmentation')
    cmd:option('-hmGauss',             1, 'Heatmap gaussian size')
    cmd:text()

    local opt = cmd:parse(arg or {})
    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.dataDir = paths.concat(opt.dataDir, 'mpii')
    opt.save = paths.concat(opt.expDir, opt.expID)
    return opt
end

-------------------------------------------------------------------------------
-- Process command line options
-------------------------------------------------------------------------------

opt = parse(arg)

if opt.dropLR == -1 then
    opt.dropLR = opt.nEpochs / 2
end


if opt.DEBUG > 2 then
    opt.trainBatch = 1
    opt.nThreads = 1
end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

if opt.GPU == -1 then
    nnlib = nn
else
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.GPU)
end

if opt.branch ~= 'none' or opt.continue then
    -- Continuing training from a prior experiment
    -- Figure out which new options have been set
    local setOpts = {}
    for i = 1,#arg do
        if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
    end

    -- Where to load the previous options/model from
    if opt.branch ~= 'none' then opt.load = opt.expDir .. '/' .. opt.branch
    else opt.load = opt.expDir .. '/' .. opt.expID end

    -- Keep previous options, except those that were manually set
    local opt_ = opt
    opt = torch.load(opt_.load .. '/options.t7')
    opt.save = opt_.save
    opt.load = opt_.load
    opt.continue = opt_.continue
    for i = 1,#setOpts do opt[setOpts[i]] = opt_[setOpts[i]] end

    epoch = opt.lastEpoch + 1
    
    -- If there's a previous optimState, load that too
    if paths.filep(opt.load .. '/optimState.t7') then
        optimState = torch.load(opt.load .. '/optimState.t7')
        optimState.learningRate = opt.LR
    end

else epoch = 1 end
opt.epochNumber = epoch

-- Track accuracy
opt.acc = {train={}, valid={}}

-- Save options to experiment directory
torch.save(opt.save .. '/options.t7', opt)

end
