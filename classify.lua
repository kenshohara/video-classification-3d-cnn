function classify_video(video_dir, video_name)
  cutorch.synchronize()

  local files = {}
  for x in lfs.dir(video_dir) do
    if x ~= '.' and x ~= '..' then
      table.insert(files, x)
    end
  end
  local n_frames = #files

  local clips = {}
  for t = 1, n_frames - opt.sample_duration, opt.sample_duration do
    local sample_begin_t = t
    local sample_end_t = t + opt.sample_duration - 1
    local sample_data = {}
    sample_data.video = video_dir
    sample_data.segment = {sample_begin_t, sample_end_t}
    table.insert(clips, sample_data)
  end

  video_outputs = torch.Tensor(#clips, opt.n_classes):fill(0)
  next_clip_index = 1
  for j = 1, #clips, opt.batch_size do
    task_queue:addjob(
        function()
          collectgarbage()
          local size = math.min(opt.batch_size, #clips - j + 1)
          local inputs = torch.Tensor(size, 3, opt.sample_duration,
                                      opt.sample_size, opt.sample_size)

          local end_k = math.min((j + opt.batch_size - 1), #clips)
          for k = j, end_k do
            local video_directory_path = clips[k].video
            local begin_t = clips[k].segment[1]
            local end_t = clips[k].segment[2]

            local sample = data_loader.load_center_sample(
                video_directory_path, opt.sample_size, begin_t, end_t)

            inputs[k - j + 1] = sample
          end

          collectgarbage()

          return inputs
        end,
        classify_batch
    )
  end

  task_queue:synchronize()
  cutorch.synchronize()

  results = {}
  for i = 1, #clips do
    clip_results = {}
    table.insert(clip_results, video_name)
    table.insert(clip_results, clips[i].segment[1])
    table.insert(clip_results, clips[i].segment[2])
    for j = 1, opt.n_classes do
      table.insert(clip_results, video_outputs[i][j])
    end
    table.insert(results, clip_results)
  end

  return results
end

local inputs
inputs = torch.CudaTensor()

function classify_batch(inputs_cpu)
  local batch_size = inputs_cpu:size(1)
  if batch_size < 10 then
    local new_size = inputs_cpu:size()
    new_size[1] = new_size[1] * 2
    inputs_cpu = inputs_cpu:resize(new_size)
    inputs_cpu[{{batch_size + 1, new_size[1]}, {}, {}, {}}] =
        inputs_cpu[{{1, batch_size}, {}, {}, {}, {}}]
  end

  inputs:resize(inputs_cpu:size()):copy(inputs_cpu)

  local outputs = model:forward(inputs)
  if outputs:dim() == 1 then
    outputs = outputs:reshape(inputs:size(1), outputs:size(1) / inputs:size(1))
  end
  cutorch.synchronize()

  outputs = outputs:float()
  for i = 1, batch_size do
    local index = next_clip_index + i - 1
    video_outputs[index] = outputs[i]
  end
  next_clip_index = next_clip_index + batch_size
end
