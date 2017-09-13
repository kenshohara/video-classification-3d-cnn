require('torch')
require('nn')
require('lfs')

require('cutorch')
require('cunn')
require('cudnn')
cudnn.fastest = true
cudnn.benchmark = true
cudnn.verbose = false

opts = dofile('opts.lua')
opt = opts.parse(arg)

opt.sample_size = 112
opt.sample_duration = 16
opt.n_classes = 400

model = torch.load(opt.model)
-- if #model == 1 then -- remove outer nn.sequential
--   model = model:get(1)
-- end
-- for i = 1, #model do
--   local weight = model:get(i).weight
--   local gradWeight = model:get(i).gradWeight
--   if weight and gradWeight and gradWeight:size():size() == 0 then -- if gradWeight is empty
--     model:get(i).gradWeight = weight:clone()
--     model:get(i).gradBias = model:get(i).bias:clone()
--   end
-- end

threads = require('threads')
do
  local options = opt
  task_queue = threads.Threads(
      opt.n_threads,
      function()
        require('torch')
        require('image')
        data_loader = dofile('data_loader.lua')
        mean = {114.7748, 107.7354, 99.4750}
      end,
      function(thread_id)
        opt = options
        id = thread_id
      end
  )
end

local fin = io.open(opt.input, 'r')
local input_files = {}
for line in fin:lines() do
  table.insert(input_files, line)
end
fin:close()

dofile('classify.lua')

local output = {}
os.execute('mkdir tmp')
for i = 1, #input_files do
  if paths.filep(input_files[i]) then
    local video_dir = string.format('tmp/%s', input_files[i])
    os.execute(string.format('mkdir %s', video_dir))
    os.execute(string.format('ffmpeg -i %s %s/image_%%05d.jpg',
                             input_files[i], video_dir))

    local results = classify_video(video_dir, input_files[i])
    for j = 1, #results do
      table.insert(output ,results[j])
    end

    os.execute(string.format('rm -rf tmp/%s', input_files[i]))
  else
    print(string.format('%s does not exist', inputs_files[i]))
  end
end

local fout = io.open(opt.output, 'w')
for i = 1, #output do
  for j = 1, #output[i] - 1 do
    fout:write(output[i][j])
    fout:write(',')
  end
  fout:write(output[i][#output[i]])
end
fout:close()
