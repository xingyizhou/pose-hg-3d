-- Main processing step
function step(tag, tpValid)
    loader[tag].tpValid = tpValid
    if tpValid then
        print('valid on ' .. tpValid)
    end
    local avgLoss, avgAcc, mpje, avgVarLoss2D, avgRegLoss3D, correct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end
    local batch, nIters
    
    if tag == 'train' then
        model:training()
        set = 'train'
        batch = opt.trainBatch
        nIters = opt.trainIters
    else
        batch = opt.validBatch
        model:evaluate()
        set = 'valid'
        nIters = opt.validIters[tpValid]
    end

    local tmpVarCriterion = VarCriterion()
    local tmpCriterion = nn.MSECriterion()
    
    local nSamples2D, nSamples3D = 0, 0
    local tmp = tpValid or 'train'
    
    for i,sample in loader[set]:run(tmp) do
        xlua.progress(i, nIters)
        local input, label, indices = unpack(sample)

        if opt.GPU ~= -1 then
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end

        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        local len = output[#output]:size(1)
        
        for t = 1, len do 
            local z = output[#output]:narrow(1, t, 1)
            local p = getPreds(label[1]:narrow(1, t, 1))
            local lb = (tag == 'train' and getPreds(label[1]:narrow(1, t, 1))) or getPreds(output[#output - 1]:narrow(1, t, 1)) 
            local vis = torch.ones(z:size(1), z:size(2), 1)
            local tmpOutput = output[#output - 1]:narrow(1, t, 1)
            for k = 1,p:size(1) do
                for j = 1,p:size(2) do 
                    if lb[k][j][1] == 1 and lb[k][j][2] == 1 then
                        vis[k][j][1] = 0
                    end
                    local hm = tmpOutput[k][j]
                    local pX,pY = p[k][j][1], p[k][j][2]
                    if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
                       local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                       p[k][j]:add(diff:sign():mul(.25))
                    end
                end
            end
            p:add(0.5)
            p = p / opt.outputRes * 2 - 1
            p = torch.cat(p, vis, 3)
            
            p:cuda()
            label[#label][2][t] = p
            
            local labelzsum = label[#label][1][t]:sum()
            local tp = tpValid or (((labelzsum < - dataset.nJoints + opt.epsilon and labelzsum > - dataset.nJoints - opt.epsilon) and 'mpii') or 'h36m')
            if tp == 'mpii' then
                nSamples2D = nSamples2D + 1
                avgVarLoss2D = avgVarLoss2D + tmpVarCriterion:forward(z, p)
            elseif tp == 'h36m' then
                nSamples3D = nSamples3D + 1
                avgRegLoss3D = avgRegLoss3D + tmpCriterion:forward(output[#output]:narrow(1, t, 1), label[#label][1]:narrow(1, t, 1))
            else
                nSamples3D = nSamples3D + 1
            end
            
            local pje_iter, correct_iter = eval(indices:narrow(1, t, 1), output[#output]:narrow(1, t, 1), output[#output - 1]:narrow(1, t, 1), tp, img)
            mpje = mpje + pje_iter 
            correct = correct + correct_iter
        end
        
        if not (tpValid == 'mpi_inf_3dhp') then
            avgLoss = avgLoss + criterion:forward(output, label) * output[1]:size(1)
            avgAcc = avgAcc + accuracy(output, label) * output[1]:size(1)
        end
        
        if tag == 'train' then
            model:zeroGradParameters()
            model:backward(input, criterion:backward(output, label))
            optfn(evalFn, param, optimState)
        end

        local nSamples = nSamples2D + nSamples3D
        if opt.DEBUG > 0 and (i % opt.display == 0 or i == 1) then
            print('Iter %d, err %.7f, Var2D %.4f, Reg %.4f, Acc %.4f, MPJE %.4f, PCK %.4f' % {i, avgLoss / nSamples, avgVarLoss2D / nSamples2D, avgRegLoss3D / nSamples3D, avgAcc / nSamples, mpje / nSamples3D, correct . nSamples})
        end
    end
    
    local nSamples = nSamples2D + nSamples3D
    avgLoss = avgLoss / nSamples
    avgAcc = avgAcc / nSamples
    mpje = mpje / nSamples3D
    correct = correct / nSamples
    avgVarLoss2D = avgVarLoss2D / nSamples2D
    avgRegLoss3D = avgRegLoss3D / nSamples3D
    print(string.format("      %s : Loss: %.7f Var2D: %.4f Reg3D: %.7f Acc: %.4f MPJE: %.4f, PCK %.4f"  % {set, avgLoss, avgVarLoss2D, avgRegLoss3D, avgAcc, mpje, correct}))
    
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['PCK       '] = string.format("%.4f" % correct),
            ['mpje       '] = string.format("%.4f" % mpje),
            ['Var2D       '] = string.format("%.4f" % avgVarLoss2D),
            ['Reg3D       '] = string.format("%.4f" % avgRegLoss3D),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        print('Snapshot')
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    end
    
    if (epoch % opt.dropLR == 0 and tag == 'train') then
        optimState.learningRate = optimState.learningRate / 10
        print('Drop LR to ' .. optimState.learningRate)
    end
    
end

function train() 
    if opt.trainIters > 0 then
        step('train') 
    end
end
function valid() 
    if opt.valid3DHP then
        step('valid', 'mpi_inf_3dhp') 
    end
    if opt.validH36M then
        step('valid', 'h36m') 
    end
    step('valid', 'mpii')
end
function predict() step('predict') end
function demo() step('valid', 'demo') end
