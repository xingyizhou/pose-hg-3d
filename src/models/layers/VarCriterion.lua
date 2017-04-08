local VarCriterion, parent = torch.class('VarCriterion', 'nn.Criterion')

function VarCriterion:__init()
    parent.__init(self)
    if opt.shortVar then
        self.skeletonRef = {{{1,2,1},    {2,3,1},
                             {4,5,2},    {5,6,2}},
                            {{11,12,3},  {12,13,3},
                             {14,15,4},  {15,16,4}}}
        self.skeletonWeight = {{0.662904952507, 0.657260067945, 
                                0.657260067945, 0.662904952507}, 
                               {1.19995651686, 1.05487331532, 
                                1.05487331532, 1.19995651686}}
    else
        self.skeletonRef = {{{1,2,1},    {2,3,1},
                             {4,5,2},    {5,6,2},
                             {11,12,3},  {12,13,3},
                             {14,15,4},  {15,16,4}, 
                             {3,7,1}, {4, 7, 2}, 
                             {13,9,3}, {14,9,4}}}
        self.skeletonWeight = {{0.662904952507, 0.657260067945, 
                                0.657260067945, 0.662904952507, 
                                1.19995651686, 1.05487331532, 
                                1.05487331532, 1.19995651686, 
                                2.23332837788, 2.23332837788, 
                                1.65793486969, 1.65793486969}}
    end
    
    if opt.standRatioVar then
        self.skeletonRef = {{{1,2,1},    {2,3,1},
                             {4,5,2},    {5,6,2}},
                            {{11,12,3},  {12,13,3},
                             {14,15,4},  {15,16,4}}, 
                             {{3,7,1}, {4, 7, 2}}, 
                             {{13,9,3}, {14,9,4}}}
        self.skeletonWeight = {{1.0085885098415446, 1, 
                                1, 1.0085885098415446}, 
                               {1.1375361376887123, 1, 
                                1, 1.1375361376887123}, 
                               {1, 1}, 
                               {1, 1}}
    end
    

end

function VarCriterion:updateOutput(input, target)
    local z = input
    local xy = target
    self.output = 0
    local batSize = z:size(1)
    for i = 1, batSize do 
        for group = 1, #self.skeletonRef do
            local E, num = 0, 0
            local N = #self.skeletonRef[group]
            local l = torch.zeros(N)
            for j = 1, N do 
                local id1, id2 = self.skeletonRef[group][j][1], self.skeletonRef[group][j][2]
                if xy[i][id1][3] > 0.5 and xy[i][id2][3] > 0.5 then
                    local p1 = {xy[i][id1][1], xy[i][id1][2], z[i][id1]}
                    local p2 = {xy[i][id2][1], xy[i][id2][2], z[i][id2]}
                    l[j] = torch.sqrt((p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]) + (p1[3] - p2[3]) * (p1[3] - p2[3]))
                    l[j] = l[j] * self.skeletonWeight[group][j]
                    num = num + 1
                    E = E + l[j]
                end
            end
            E = (num < 0.5 and 0.0) or (E / num)

            for j = 1, N do 
                if l[j] > 0 then
                    self.output = self.output + (l[j] - E) * (l[j] - E) / num
                end
            end
        end
    end
    
    self.output = self.output / batSize
    return self.output
end

function VarCriterion:updateGradInput(input, target)
    local z = input
    local xy = target
    self.gradInput:resizeAs(z)
    self.gradInput:fill(0)
    local batSize = z:size(1)
    local norm = batSize
    for i = 1, batSize do
        for group = 1, #self.skeletonRef do 
            local E, num = 0, 0
            local N = #self.skeletonRef[group]
            local l = torch.zeros(N) 
            for j = 1, N do
                local id1, id2 = self.skeletonRef[group][j][1], self.skeletonRef[group][j][2]
                if xy[i][id1][3] > 0.5 and xy[i][id2][3] > 0.5 then
                    local p1 = {xy[i][id1][1], xy[i][id1][2], z[i][id1]}
                    local p2 = {xy[i][id2][1], xy[i][id2][2], z[i][id2]}
                    l[j] = torch.sqrt((p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]) + (p1[3] - p2[3]) * (p1[3] - p2[3]))
                    l[j] = l[j] * self.skeletonWeight[group][j]
                    num = num + 1
                    E = E + l[j]
                end
            end
            E = (num < 0.5 and 0.0) or (E / num)
            
            for j = 1, N do 
                if l[j] > 0 then
                    local id1, id2 = self.skeletonRef[group][j][1], self.skeletonRef[group][j][2]
                    self.gradInput[i][id1] = self.gradInput[i][id1] + 2 * self.skeletonWeight[group][j] * self.skeletonWeight[group][j] / num * 
                                                                      (l[j] - E) / l[j] * (z[i][id1] - z[i][id2]) / norm
                    self.gradInput[i][id2] = self.gradInput[i][id2] + 2 * self.skeletonWeight[group][j] * self.skeletonWeight[group][j] / num * 
                                                                      (l[j] - E) / l[j] * (z[i][id2] - z[i][id1]) / norm
                end
            end
        end
    end
    return self.gradInput
end

