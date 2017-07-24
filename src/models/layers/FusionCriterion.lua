local FusionCriterion, parent = torch.class('FusionCriterion', 'nn.Criterion')
paths.dofile('VarCriterion.lua')
function FusionCriterion:__init(regWeight, varWeight)
   parent.__init(self)
   self.regWeight = regWeight
   self.varWeight = varWeight
   self.mseCriterion = nn.MSECriterion()
   self.varCriterion = VarCriterion()
end

function FusionCriterion:updateOutput(input, target)
    local batSize = input:size(1)
    self.output = 0
    local regTarget = target[1]
    local weakTarget = target[2]
    for t = 1, batSize do 
        if (regTarget[t]:sum() < - dataset.nJoints + opt.epsilon and regTarget[t]:sum() > - dataset.nJoints - opt.epsilon) then
            self.output = self.varWeight * self.varCriterion:updateOutput(input:narrow(1, t, 1), weakTarget:narrow(1, t, 1)) 
        else
            self.output = self.regWeight * self.mseCriterion:updateOutput(input:narrow(1, t, 1), regTarget:narrow(1, t, 1))
        end
    end
    self.output = self.output / batSize
    return self.output
end

function FusionCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:fill(0)
    local batSize = input:size(1)
    local regTarget = target[1]
    local weakTarget = target[2]
    for t = 1, batSize do 
        if (regTarget[t]:sum() < - dataset.nJoints + opt.epsilon and regTarget[t]:sum() > - dataset.nJoints - opt.epsilon) then
            self.gradInput[t] = self.varWeight * self.varCriterion:updateGradInput(input:narrow(1, t, 1), weakTarget:narrow(1, t, 1))
        else
            self.gradInput[t] = self.regWeight * self.mseCriterion:updateGradInput(input:narrow(1, t, 1), regTarget:narrow(1, t, 1))
        end
    end
    self.gradInput = self.gradInput / batSize
    return self.gradInput
end

