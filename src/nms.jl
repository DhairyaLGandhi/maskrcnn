function nms(boxes_with_scores, threshold)
  scores = boxes_with_scores[:,5]
  boxes = boxes_with_scores[:, 1:4]
  y1 = boxes[:,1]
  x1 = boxes[:,2]
  y2 = boxes[:,3]
  x2 = boxes[:,4]
  areas = (x2 .- x1 .+ 1) .* (y2 .- y1 .+ 1) # + 1
  idxs = sortperm(scores, rev = true)
  # @show idxs[1:10]

  # sort boxes based on scores
  # @show scores[1:10], scores[end-10:end]
  boxes = boxes[idxs,:]
  pick = Int[]
  while length(idxs) > 0
    las = length(idxs) # - 1
    i = idxs[las]
    push!(pick, i)

    xx1 = max.(x1[i], x1[idxs[1:las]])
    xx2 = max.(x2[i], x2[idxs[1:las]])
    yy1 = max.(y1[i], y1[idxs[1:las]])
    yy2 = max.(y2[i], y2[idxs[1:las]])

    w = max.(0.0f0, xx2 .- xx1 .+ 1)
    h = max.(0.0f0, yy2 .- yy1 .+ 1)
    # ff = findall(isnan, w)
    # if length(ff) > 0
    #   @show "nans in w"
    #   @show "!!!!!!!!!"
    # end
    overlap = (w .* h) ./ areas[idxs[1:las]]
    # @show overlap[1:10]
    inds = findall(x -> x > threshold, overlap)
    deleteat!(idxs, unique([inds..., las]))
  end
  # boxes[pick, :]
  bs = boxes[pick, 1:4]
  as = (bs[:,3] .- bs[:,1] .+ 1) .* (bs[:,4] .- bs[:,2] .+ 1)
  @show as
  pick
end

function nms2(boxes_with_scores, threshold)
  x1 = boxes_with_scores[:, 2]
    y1 = boxes_with_scores[:, 1]
    x2 = boxes_with_scores[:, 4]
    y2 = boxes_with_scores[:, 3]
    scores = boxes_with_scores[:, 5]

    areas = @. (x2 - x1 + 1) * (y2 - y1 + 1)
    @info "Mean of areas: $(mean(areas))"
    @info "sample boxes: $(boxes_with_scores[1,:])"

    # sorting is inefficient on the GPU
    # moving the actual thing off GPU
    # breaks tracking 
    s = copy(scores) |> cpu
    order = sortperm(s, rev = true)
    @show "after order"

    keep = 1:size(boxes_with_scores, 1) |> collect # need to setindex in keep
    num_out = 0

    supp = _nms!(keep, num_out, boxes_with_scores, order, areas, threshold)

    # keep[1:num_out]
    supp
end

function _nms!(keep_out, num_out, boxes, order, areas, nms_overlap_thresh)
  boxes_num = size(boxes, 1)
  boxes_dim = size(boxes, 2)

  keep_out_flat = keep_out
  @show typeof(boxes)
  boxes_flat = boxes
  order_flat = order
  areas_flat = areas

  suppressed = zeros(boxes_num)

  num_to_keep = 1
  for _i = 1:boxes_num
    i = order_flat[_i]
    suppressed[i] == 1 && continue
    keep_out_flat[num_to_keep] = i
    num_to_keep += 1

    ix1 = boxes_flat[i, 1]
    iy1 = boxes_flat[i, 2]
    ix2 = boxes_flat[i, 3]
    iy2 = boxes_flat[i, 4]

    iarea = areas_flat[i]
    
    # _j = collect((_i+1):boxes_num)
    # j = order_flat[_j]
    # xx1 = max.(ix1, boxes_flat[j,1])
    # yy1 = max.(iy1, boxes_flat[j,2])
    # xx2 = min.(ix2, boxes_flat[j,3])
    # yy2 = min.(iy2, boxes_flat[j,4])
    # w = max.(0.0f0, xx2 .- xx1 .+ 1.f0)
    # h = max(0.0f0, yy2 .- yy1 .+ 1.f0)
    # inter = w .* h
    # ovr = inter ./ (iarea .+ areas_flat[j] .- inter)
    # iis = ovr .>= nms_overlap_thresh
    # suppressed
    for _j = (_i+1):boxes_num
      j = order_flat[_j]
      suppressed[j] == 1 && continue

      xx1 = max(ix1, boxes_flat[j,1])
      yy1 = max(iy1, boxes_flat[j,2])
      xx2 = min(ix2, boxes_flat[j,3])
      yy2 = min(iy2, boxes_flat[j,4])
      w = max(0.0f0, xx2 - xx1 + 1)
            h = max(0.0f0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas_flat[j] - inter)
            # @show ovr
            if ovr >= nms_overlap_thresh
              suppressed[j] = 1
            end
        end
    end

    num_out = num_to_keep
    num_out = min(num_out, length(keep_out_flat))
    keep_out_flat[1:num_out]
end


DIVUP(m,n) = ((m) / (n) + ((m) % (n) > 0))

function iou(box1, box2)
    left = max(box1[1], box2[1])
    top = max(box1[2], box2[2])
    right = min(box1[3], box2[3])
    bottom = min(box1[4], box2[4])

    width = max(right - left + 1.f0, 0.f0)
    height = max(bottom - top + 1.f0, 0.f0)

    interS = width * height
    S1 = (box1[3] - box1[1] + 1.f0) * (box1[4] - box1[2] + 1.f0)
    S2 = (box2[3] - box2[1] + 1.f0) * (box2[4] - box2[2] +1.f0)
    interS / (S1 + S2 - interS)
end

function nms_kernel!(block_boxes, n_boxes,nms_overlap_thresh, dev_boxes, dev_mask)
    row_start = blockIdx().y
    col_start = blockIdx().x

    row_start > col_start && return

    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)

    # block_boxes = cu(zeros(n_boxes, 5))

    if threadIdx().x < col_size
        block_boxes[(threadIdx().x-1) * 5 + 1-1] =
            dev_boxes[threadsPerBlock * col_start + threadIdx().x + 0]
        block_boxes[(threadIdx().x-1) * 5 + 2-1] =
            dev_boxes[threadsPerBlock * col_start + threadIdx().x + 1]
        block_boxes[(threadIdx().x-1) * 5 + 3-1] =
            dev_boxes[threadsPerBlock * col_start + threadIdx().x + 2]
        block_boxes[(threadIdx().x-1) * 5 + 4-1] =
            dev_boxes[threadsPerBlock * col_start + threadIdx().x + 3]
        block_boxes[(threadIdx().x-1) * 5 + 5-1] =
            dev_boxes[threadsPerBlock * col_start + threadIdx().x + 4]
    end

    sync_threads()
    # return nothing
    if threadIdx().x < row_size
        cur_box_idx = threadsPerBlock * row_start + threadIdx().x
        cur_box = dev_boxes[cur_box_idx, :]
        i = 0
        t = 0.f0
        start = 0
        if row_start == col_start
          start = threadIdx().x + 1
        end
        for i = start:col_size
          if iou(cur_box, block_boxes[i,:]) > nms_overlap_thresh
            t |= 1 << i
          end
        end
        col_blocks = DIVUP(n_boxes, threadsPerBlock)
        dev_mask[cur_box_idx * col_blocks + col_start] = t
    end
    return nothing
end

function _nms_kernel!(block_boxes, boxes_num, boxes_flat, mask_flat, thresh)
    @cuda blocks = DIVUP(boxes_num, threadsPerBlock), threads = threadsPerBlock nms_kernel!(block_boxes, boxes_num, thresh, boxes_flat, mask_flat)
end

function cuda_nms(dets, thresh)
    x1 = dets[:, 2]
    y1 = dets[:, 1]
    x2 = dets[:, 4]
    y2 = dets[:, 3]
    scores = dets[:, 5]

    dets_temp = similar(dets)
    dets_temp[:, 1] = dets[:, 2]
    dets_temp[:, 2] = dets[:, 1]
    dets_temp[:, 3] = dets[:, 4]
    dets_temp[:, 4] = dets[:, 3]
    dets_temp[:, 5] = dets[:, 5]

    areas = @. (x2 - x1 + 1) * (y2 - y1 + 1)
    s = copy(scores) |> cpu
    order = sortperm(scores, rev = true)
    dets = dets[order, :]

    keep = zeros(Float32, size(dets, 1))
    num_out = 0

    boxes_num = size(dets_temp, 1)
    boxes_dim = size(dets_temp, 2)

    boxes_flat = dets_temp
    col_blocks = DIVUP(boxes_num, threadsPerBlock) |> round

    mask_flat = zeros(boxes_num, col_blocks) |> gpu

    block_boxes = cu(zeros(boxes_num, 5))

    _nms_kernel!(block_boxes, boxes_num, boxes_flat, mask_flat, thresh)

    mask_cpu_flat = zeros(Float32, boxes_num, col_blocks)
    remv_cpu = zeros(Float32, col_blocks)

    keep_flat = keep
    num_to_keep = 0

    # for i in 1:boxes_num
    #     nblock = i / threadsPerBlock
    #     inblock = i % threadsPerBlock

    #     if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
    #         keep_flat[num_to_keep++] = i;
    #         p = mask_cpu_flat[ + i * col_blocks;
    #         for j = nblock:col_blocks
    #             remv_cpu_flat[j] |= p[j];
    #         end
    #     end
    # end
end