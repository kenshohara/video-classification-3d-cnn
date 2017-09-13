local utils = {}

function utils.make_data_parallel(model, first_gpu_id, n_gpus)
  if n_gpus < 2 then
    return model
  end

  assert(n_gpus <= cutorch.getDeviceCount(), 'number of GPUs less than n_gpus specified')
  local gpu_table = torch.range(first_gpu_id, first_gpu_id + n_gpus - 1):totable()
	local fastest, benchmark = cudnn.fastest, cudnn.benchmark
  local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(
    function()
      require 'cudnn'
      cudnn.fastest = fastest
      cudnn.benchmark = benchmark
    end)
  dpt.gradInput = nil
  model = dpt:cuda()

  return model
end

function utils.get_cropping_box(box_width, box_height, image_width, image_height, position)
  if position == 'c' then
    local center_x = math.floor(image_width / 2)
    local center_y = math.floor(image_height / 2)
    local box_half_width = math.floor(box_width / 2)
    local box_half_height = math.floor(box_height / 2)
    return center_x - box_half_width + 1, center_y - box_half_height + 1,
        center_x + box_half_width, center_y + box_half_height
  elseif position == 'tl' then
    return 1, 1, box_width, box_height
  elseif position == 'tr' then
    return image_width - box_width + 1, 1, image_width, box_height
  elseif position == 'bl' then
    return 1, image_height - box_height + 1, box_width, image_height
  elseif position == 'br' then
    return image_width - box_width + 1, image_height - box_height + 1, image_width, image_height
  end

  return 1, 1, 1, 1
end

return utils
