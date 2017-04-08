require 'os'
require 'hdf5'
local projectDir = '/home/zxy/Projects/pose-hg-3d'
local pyFile = projectDir .. '/src/util/pyTools.py'


function saveData(dict, tmpFile)
    local file = hdf5.open(tmpFile, 'w')
    for k, v in pairs(dict) do 
        file:write(k, v)
    end
    file:close()
end

function pyFunc(func, data) -- data is a dict
    local tmpFile = 'tmp/' .. torch.random() ..  '.h5'
    if io.open(tmpFile) then os.execute('rm ' .. tmpFile) end
    saveData(data, tmpFile)
    os.execute('python ' .. pyFile .. ' ' .. func .. ' ' .. tmpFile)
    os.execute('rm ' .. tmpFile)
end


