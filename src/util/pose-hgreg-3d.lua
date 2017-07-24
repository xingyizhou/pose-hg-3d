-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nJoints,5}
ref.outputDim = {}
paths.dofile('../models/layers/FusionCriterion.lua')

criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
    criterion:add(nn[opt.crit .. 'Criterion']())
end
FusionCriterion = FusionCriterion(opt.regWeight, opt.varWeight)
criterion:add(FusionCriterion)

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images
function generateSample(set, idx, tp)
    if tp == 'h36m' then -- load 3D data
        local img = dataset:loadImage(idx, tp)
        local pts, c, s, pts_3d = dataset:getPartInfo(idx, tp)
        local r = 0
        -- Fix torsor annotation discripency between h36m and mpii
        pts_3d[8] = (pts_3d[13] + pts_3d[14]) / 2
        local inp = crop(img, c, s * 224.0 / 200.0, r, opt.inputRes)
        local outMap = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
        local outReg = torch.zeros(dataset.nJoints, 1)
        
        for i = 1,dataset.nJoints do
            local pt = transform3DFloat(pts_3d[i], c, s * dataset.h36mImgSize / 200.0, r, opt.outputRes)
            local pt_2d = {pt[1], pt[2]}
            if pts[i][1] > 1 then 
                drawGaussian(outMap[i], pt_2d, opt.hmGauss)
            end
            outReg[i][1] = pt[3] / opt.outputRes * 2 - 1
        end

        return inp, outMap, outReg
    elseif tp == 'mpii' then
        local img = dataset:loadImage(idx, tp)
        local pts, c, s = dataset:getPartInfo(idx, tp)
        local r = 0

        if set == 'train' then
            s = s * (2 ^ rnd(opt.scale))
            r = rnd(opt.rotate)
            if torch.uniform() <= .6 then r = 0 end
        end

        local inp = crop(img, c, s, r, opt.inputRes)
        local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)
        for i = 1,dataset.nJoints do
            if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
                drawGaussian(out[i], transform(pts[i], c, s, r, opt.outputRes), opt.hmGauss)
            end
        end

        if set == 'train' then
            if torch.uniform() < .5 then
                inp = flip(inp)
                out = shuffleLR(flip(out))
            end
            inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
            inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
            inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        end
        return inp,out
    elseif tp == 'mpi_inf_3dhp' then
        local img = dataset:loadImage(idx, tp)
        local  c = torch.ones(2) * dataset.mpiinf3dhpImgSize / 2
        local inp = crop(img, c, 1 * dataset.mpiinf3dhpImgSize / 200.0, 0, opt.inputRes) -- 200.0 is the default bbox size in MPII dataset
        return inp
    elseif tp == 'demo' then
        local img = dataset:loadImage(idx, tp):narrow(1, 1, 3)
        local h, w = img:size(2), img:size(3)
        local c = torch.Tensor({w / 2, h / 2})
        local size = math.max(h, w)
        local inp = crop(img, c, 1 * size / 200.0, 0, opt.inputRes)
        return inp
    end
    
end

-- Load in a mini-batch of data
function loadData(set, idxs, tps)
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    local nsamples = idxs:size(1)
    local input,labelMap, labelReg
    for i = 1, nsamples do 
        tp = tps[i]
        local tmpInput, tmpLabelMap, tmpLabelReg
        if tp == 'h36m' then
            tmpInput,tmpLabelMap, tmpLabelReg = generateSample(set, idxs[i], tp)
        elseif tp == 'mpii' then
            tmpInput,tmpLabelMap = generateSample(set, idxs[i], tp)
            tmpLabelReg = torch.ones(dataset.nJoints, 1) * -1
        elseif tp == 'mpi_inf_3dhp' or tp == 'demo' then
            tmpInput = generateSample(set, idxs[i], tp)
            tmpLabelMap = torch.ones(dataset.nJoints, opt.outputRes, opt.outputRes) * -1
            tmpLabelReg = torch.zeros(dataset.nJoints, 1) * -1
        end    
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabelMap = tmpLabelMap:view(1,unpack(tmpLabelMap:size():totable()))
        tmpLabelReg = tmpLabelReg:view(1,unpack(tmpLabelReg:size():totable()))
        if not input then
            input = tmpInput
            labelMap = tmpLabelMap
            labelReg = tmpLabelReg
        else
            input = input:cat(tmpInput,1)
            labelMap = labelMap:cat(tmpLabelMap, 1)
            labelReg = labelReg:cat(tmpLabelReg, 1)
        end
    end
    local newLabel = {}
    for i = 1,opt.nStack do newLabel[i] = labelMap end
    newLabel[opt.nStack + 1] = {labelReg, torch.ones(nsamples, dataset.nJoints, 3) * -1}
    return input,newLabel
end

function accuracy(output,label)
    return heatmapAccuracy(output[#output - 1],label[#label - 1],nil,dataset.accIdxs)
end

function eval(idx, outputReg, outputHM, tp, img)
    local Reg = outputReg
    local tmpOutput = outputHM
    Reg = Reg:view(Reg:size(1), 16, 1)
    local z = (Reg + 1) * opt.outputRes / 2 
    
    local p = getPreds(tmpOutput)
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    local dis = 0
    local correct = p:size(1)
    for i = 1,p:size(1) do
        if tp == 'mpii' or tp == 'demo' then
            local pred = torch.zeros(dataset.nJoints, 3)
            for j = 1, dataset.nJoints do
              pred[j][1], pred[j][2], pred[j][3] = p[i][j][1], p[i][j][2], z[i][j][1]
            end
            pred = pred * 4
            if opt.DEBUG > 2 then
                pyFunc('Show3d', {joint=pred, img=img, noPause = torch.zeros(1), id = torch.ones(1) * imgid})
            end
        elseif tp == 'h36m' then
            local _, c_, s_, gt_3d_ = dataset:getPartInfo(idx[i], tp)
            local gt_3d, scale, c2d, c3d, c, s, action = dataset:getPartDetail(idx[i])
            local pred = torch.zeros(dataset.nJoints, 3)
            local predVis = torch.zeros(dataset.nJoints, 3)
            local gtVis = torch.zeros(dataset.nJoints, 3)

            for j = 1, dataset.nJoints do
                pred[j][1], pred[j][2], pred[j][3] = p[i][j][1], p[i][j][2], z[i][j][1]
            end
                
            local len_pred = 0
            local len_gt = 0
            for j = 1, dataset.nBones do
                len_pred = len_pred +  ((pred[dataset.skeletonRef[j][1]][1] - pred[dataset.skeletonRef[j][2]][1]) ^ 2 + 
                                        (pred[dataset.skeletonRef[j][1]][2] - pred[dataset.skeletonRef[j][2]][2]) ^ 2 + 
                                        (pred[dataset.skeletonRef[j][1]][3] - pred[dataset.skeletonRef[j][2]][3]) ^ 2) ^ 0.5
            end
             
            len_gt = 4296.99233013
            local root = 7
            local proot = pred[root]:clone()
            for j = 1, dataset.nJoints do
                pred[j][1] = (pred[j][1] - proot[1]) / len_pred * len_gt + gt_3d[root][1]
                pred[j][2] = (pred[j][2] - proot[2]) / len_pred * len_gt + gt_3d[root][2]
                pred[j][3] = (pred[j][3] - proot[3]) / len_pred * len_gt + gt_3d[root][3]
            end
            
            -- Fix torsor annotation discripency between h36m and mpii
            pred[8] = (pred[7] + pred[9]) / 2
            
            for j = 1, dataset.nJoints do
                local JDis = ((pred[j][1] - gt_3d[j][1]) * (pred[j][1] - gt_3d[j][1]) + 
                       (pred[j][2] - gt_3d[j][2]) * (pred[j][2] - gt_3d[j][2]) + 
                       (pred[j][3] - gt_3d[j][3]) * (pred[j][3] - gt_3d[j][3])) ^ 0.5
                if JDis > opt.PCK_Threshold and not (j == 7 or j == 8) then
                    correct = correct - 1./14
                end
                dis = dis + JDis / p:size(1)
            end 
            
            predVis = predVis * 4
            gtVis = gtVis * 4
            if opt.DEBUG > 2 then
                pyFunc('Show3d', {gt=gtVis, joint=predVis, img=img, noPause = torch.zeros(1), id = torch.ones(1) * idx[i]})
            end
        elseif tp == 'mpi_inf_3dhp' then
            local c, s, gt_3d, action, bbox, gt_2d = dataset:getPartInfo(idx[i], tp)
            local pred = torch.zeros(dataset.nJoints, 3)
            local predVis = torch.zeros(dataset.nJoints, 3)
            local gtVis = torch.zeros(dataset.nJoints, 3)
            for j = 1, dataset.nJoints do
                pred[j][1], pred[j][2], pred[j][3] = p[i][j][1], p[i][j][2], z[i][j][1]
            end
            
            local shift = (pred[9] - pred[7]) * 0.2
            pred[7] = pred[7] + shift
            pred[3] = pred[3] + shift
            pred[4] = pred[4] + shift
            
            pred[8] = pred[7] + (pred[9] - pred[7]) / 2
            local s_pred, s_gt = 0, 0
            
            for j = 1, dataset.nBones do
                s_pred = s_pred +  ((pred[dataset.skeletonRef[j][1]][1] - pred[dataset.skeletonRef[j][2]][1]) ^ 2 + 
                                    (pred[dataset.skeletonRef[j][1]][2] - pred[dataset.skeletonRef[j][2]][2]) ^ 2 + 
                                    (pred[dataset.skeletonRef[j][1]][3] - pred[dataset.skeletonRef[j][2]][3]) ^ 2) ^ 0.5
                s_gt = s_gt +  ((gt_3d[dataset.skeletonRef[j][1]][1] - gt_3d[dataset.skeletonRef[j][2]][1]) ^ 2 + 
                                (gt_3d[dataset.skeletonRef[j][1]][2] - gt_3d[dataset.skeletonRef[j][2]][2]) ^ 2 + 
                                (gt_3d[dataset.skeletonRef[j][1]][3] - gt_3d[dataset.skeletonRef[j][2]][3]) ^ 2) ^ 0.5
            end
            
            local root = 7
            local proot = pred[root]:clone()
            local scale = s_pred / s_gt
            -- print('scale', scale)
            for j = 1, dataset.nJoints do
                predVis[j][1], predVis[j][2], predVis[j][3] = pred[j][1], pred[j][2], pred[j][3]
                gtVis[j] = (gt_3d[j] - gt_3d[root]) * scale + proot
                pred[j] = (pred[j] - proot) / scale + gt_3d[root]
            end

            local curDis = 0
            for j = 1, dataset.nJoints do
                local JDis = ((pred[j][1] - gt_3d[j][1]) * (pred[j][1] - gt_3d[j][1]) + 
                       (pred[j][2] - gt_3d[j][2]) * (pred[j][2] - gt_3d[j][2]) + 
                       (pred[j][3] - gt_3d[j][3]) * (pred[j][3] - gt_3d[j][3])) ^ 0.5
                if JDis > opt.PCK_Threshold and not (j == 7 or j == 8) then
                    correct = correct - 1./ dataset.mpi_inf_3dhp_num_joint
                end
                curDis =  JDis / p:size(1) + curDis
            end 

            dis = dis + curDis
            predVis = predVis * 4
            gtVis = gtVis * 4
            if opt.DEBUG > 2 and idx[i] % 40 == 1 then
                pyFunc('Show3d', {gt=gtVis, joint=predVis, img=img, noPause = torch.zeros(1), id = torch.ones(1) * idx[i]})
            end
        end
    end
    return dis / dataset.nJoints, correct 
end

