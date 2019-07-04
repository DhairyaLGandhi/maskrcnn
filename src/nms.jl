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