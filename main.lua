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
utils = dofile('utils.lua')
model = utils.make_data_parallel(model, opt.gpu_id, opt.n_gpus)
if opt.verbose then
  print(model)
end

threads = require('threads')
do
  local options = opt
  task_queue = threads.Threads(
      opt.n_threads,
      function()
        require('torch')
        require('image')
        utils = dofile('utils.lua')
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

local fin2 = io.open('class_names_list', 'r')
local class_names = {}
for line in fin2:lines() do
  table.insert(class_names, line)
end
fin2:close()

dofile('classify.lua')

local output = {}
if paths.dirp('tmp') then
  os.execute('rm -rf tmp')
end
os.execute('mkdir tmp')
local ffmpeg_loglevel = 'quiet'
if opt.verbose then
  ffmpeg_loglevel = 'info'
end
for i = 1, #input_files do
  if paths.filep(input_files[i]) then
    print(input_files[i])
    local videoname = string.match(input_files[i], '([^\\/]-%.?([^%.\\/]*))$')
    local video_dir = string.format('tmp/%s', videoname)
    os.execute(string.format('mkdir %s', video_dir))
    os.execute(string.format('ffmpeg -i "%s" -loglevel %s "%s/image_%%05d.jpg"',
                             input_files[i], ffmpeg_loglevel, video_dir))

    local results = classify_video(video_dir, input_files[i], class_names)
    for j = 1, #results do
      table.insert(output ,results[j])
    end

    os.execute(string.format('rm -rf %s', video_dir))
  else
    print(string.format('%s does not exist', input_files[i]))
  end
end
os.execute('rm -rf tmp')

local fout = io.open(opt.output, 'w')
for i = 1, #output do
  for j = 1, #output[i] - 1 do
    fout:write(output[i][j])
    fout:write(',')
  end
  fout:write(output[i][#output[i]])
  fout:write('\n')
end
fout:close()
