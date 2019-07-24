using Flux, Flux.Tracker, Flux.Optimise
using BSON: @load, @save
using Images
using Flux: @treelike
import StatsBase.predict
include("dataset.jl")

# use Resnet from Metalhead
struct ConvBlock
    convlayer
    norm
    nonlinearity
end

@treelike ConvBlock

function ConvBlock(kernel, chs; stride = (1, 1), pad = (0, 0))
  stride = stride
  ConvBlock(Conv(kernel, chs, stride = stride, pad = pad) |> gpu,

  BatchNorm(chs[2]) |> gpu,

  x -> relu.(x))
end

(c::ConvBlock)(x) = c.nonlinearity(c.norm(c.convlayer(x)))
# (c::ConvBlock)(x) = c.norm(c.convlayer(x))

"""
For Resnet (do it anyway), identity shortcut.
"""
struct IDBlock
    path_1
    shortcut
end

@treelike IDBlock

function IDBlock(kernel, filters)
    path_1 = Chain(ConvBlock((1,1), 3=>filters[1]),
                ConvBlock(kernel, filters[1]=>filters[2]),
                Conv((1,1), filters[2]=>filters[3]),
                BatchNorm(filters[3])) |> gpu

    IDBlock(path_1)
end

function (c::IDBlock)(x)
    op1 = c.path_1(x)
    relu.(op1 .+ x)
end

"""
IDBlock but with a conv (and add) shortcut. Use in Resnet graph building.
"""
struct Conv_Block
    path_1
    shortcut
end

@treelike Conv_Block

function Conv_Block(kernel, filters)
    path_1 = Chain(ConvBlock((1,1), 3=>filters[1]),
		ConvBlock(kernel, filters[1]=>filters[2]),
		Conv((1,1), filters[2]=>filters[3]),
		BatchNorm(filters[3])) |> gpu

    shortcut = Chain(Conv((1,1), filters[2]=>filters[3]),
                BatchNorm(filters[3])) |> gpu

    Conv_Block(path_1, shortcut)
end

function (c::Conv_Block)(x)
    op1 = c.path_1(x)
    sc = c.shortcut(x)
    relu.(op1 .+ sc)
end

struct ROIAlign
  crop_height
  crop_width
  extrapolation_value
  transform_fpcoor
  # add params to hold
end

@treelike ROIAlign

ROIAlign(crop_height, crop_width, extrapolation_value = 0.0f0, transform_fpcoor = true) = 
  ROIAlign(crop_height, crop_width, extrapolation_value, transform_fpcoor)

function (c::ROIAlign)(feature_map, boxes, box_ind)
  x1 = boxes[:,1]
  y1 = boxes[:,2]
  x2 = boxes[:,3]
  y2 = boxes[:,4]

  img_height, img_width = Float.(size(feature_map)[1:2])

  if c.transform_fpcoor
    spacing_w = (x2 .- x1) ./ img_width
    spacing_h = (y2 .- y1) ./ img_height

    nx0 = (x1 .+ (spacing_w ./ 2.f0) .- 0.5f0) ./ (img_width .- 1.0f0)
    ny0 = (y1 .+ (spacing_h ./ 2.f0) .- 0.5f0) ./ (img_height .- 1.0f0)
    nw = spacing_w .* (c.crop_width .- 1.0f0) ./ (img_width .- 1.0f0)
    nh = spacing_h .* (c.crop_height .- 1.0f0) ./ (img_height .- 1.0f0)
    boxes = cat(ny0, nx0, ny0 .+ nh, nx0 .+ nw, dims = 2)
  
  else
    x1 = x1 ./ (img_width .- 1.0f0)
    x2 = x2 ./ (img_width .- 1.0f0)
    y1 = y1 ./ (img_height .- 1.0f0)
    y2 = y2 ./ (img_height .- 1.0f0)
    boxes = cat(y1, x1, y2, x2, dims = 2)
  end

  crop_and_resize(feature_map, boxes, box_ind, c.crop_height, c.crop_width)
end

mutable struct FPN
  C1
  C2
  C3
  C4
  C5
  P6
  P5_conv1
  P5_conv2
  P4_conv1
  P4_conv2
  P3_conv1
  P3_conv2
  P2_conv1
  P2_conv2
end

@treelike FPN

function FPN(out_channels, ps...)
  cs = ps[1:5]
  P6 = MaxPool((1,1), stride = (2,2)) |> gpu
  P5_conv1 = Conv((1,1), 2048=>out_channels) |> gpu
  P5_conv2 = Conv((3,3), out_channels=>out_channels, pad = (1,1)) |> gpu

  P4_conv1 = Conv((1,1), 1024=>out_channels) |> gpu
  P4_conv2 = Conv((3,3), out_channels=>out_channels, pad = (1,1)) |> gpu

  P3_conv1 = Conv((1,1), 512=>out_channels) |> gpu
  P3_conv2 = Conv((3,3), out_channels=>out_channels, pad = (1,1)) |> gpu

  P2_conv1 = Conv((1,1), 256=>out_channels) |> gpu
  P2_conv2 = Conv((3,3), out_channels=>out_channels, pad = (1,1)) |> gpu

  FPN(cs..., P6, P5_conv1,
      P5_conv2,
      P4_conv1,
      P4_conv2,
      P3_conv1,
      P3_conv2,
      P2_conv1,
      P2_conv2)

end

function (c::FPN)(x)
  c1_out = c.C1(x)
  c2_out = c.C2(c1_out)
  c3_out = c.C3(c2_out)
  c4_out = c.C4(c3_out)
  c5_out = c.C5(c4_out)

  p5_out = c.P5_conv1(c5_out)

  p4_out = c.P4_conv1(c4_out) + gpu(upsample(cpu(p5_out), (2,2,1,1)))
  p3_out = c.P3_conv1(c3_out) + gpu(upsample(cpu(p4_out), (2,2,1,1)))
  p2_out = c.P2_conv1(c2_out) + gpu(upsample(cpu(p3_out), (2,2,1,1)))

  p5_out = c.P5_conv2(p5_out)
  p4_out = c.P4_conv2(p4_out)
  p3_out = c.P3_conv2(p3_out)
  p2_out = c.P2_conv2(p2_out)

  p6_out = c.P6(p5_out)

  p2_out, p3_out, p4_out, p5_out, p6_out
end

mutable struct Mask
  depth
  pool_size
  image_shape
  num_classes
  chain
end

@treelike Mask

function Mask(depth, pool_size, image_shape, num_classes)
  chain = Chain(ConvBlock((3,3), 256=>256, pad = (3,3)),
                ConvBlock((3,3), 256=>256, pad = (1,1)),
                ConvBlock((3,3), 256=>256),
		ConvBlock((3,3), 256=>256),
                ConvTranspose((2,2), 256=>256, relu, stride = (2,2)),
                Conv((1,1), 256=>num_classes, σ)) |> gpu

  Mask(depth, pool_size, image_shape, num_classes, chain)
end

function (c::Mask)(x, rois)
  op = pyramid_roi_align((rois, x...), c.pool_size, c.image_shape)
  op = gpu(op)
  op = c.chain(op)
  op
end

mutable struct Classifier
  depth
  pool_size
  image_shape
  num_classes
  chain
  linear_class
  linear_bbox
end

@treelike Classifier

function Classifier(depth, pool_size, image_shape, num_classes)
  chain = Chain(ConvBlock((pool_size, pool_size), depth=>1024),
              ConvBlock((1,1), 1024=>1024)) |> gpu
  linear_class = Dense(1024, num_classes) |> gpu
  linear_bbox = Dense(1024, 4*num_classes) |> gpu

  Classifier(depth, 
            pool_size,
            image_shape,
            num_classes,
            chain,
            linear_class,
            linear_bbox)
end

function (c::Classifier)(x, rois)
  x = pyramid_roi_align((rois, x...), c.pool_size, c.image_shape)
  x = gpu(x)
  x = c.chain(x)
  x = dropdims(x, dims = (1,2))
  mrcnn_class_logits = c.linear_class(x)
  mrcnn_probs = softmax(mrcnn_class_logits)
  mrcnn_bbox = c.linear_bbox(x)
  mrcnn_bbox = reshape(mrcnn_bbox, (:, 4, size(mrcnn_bbox, ndims(mrcnn_bbox))))

  mrcnn_class_logits, mrcnn_probs, mrcnn_bbox
end

mutable struct RPN
  # anchors_per_location
  # anchor_stride
  # depth
  conv_shared
  conv_class
  conv_bbox
end

@treelike RPN

function RPN(anchors_per_location::Int, anchor_stride, depth)
  conv_shared = Conv((3,3), depth=>512, stride = anchor_stride, pad = (1,1)) |> gpu
  conv_class = Conv((1,1), 512=>2*anchors_per_location) |> gpu
  conv_bbox = Conv((1,1), 512=>4*anchors_per_location) |> gpu
  RPN(
    # anchors_per_location,
    # anchor_stride,
    # depth,
    conv_shared,
    conv_class,
    conv_bbox)
end

function (c::RPN)(x)
  x = c.conv_shared(x)

  rpn_class_logits = c.conv_class(x)
  rpn_class_logits = permutedims(rpn_class_logits, (3,4,2,1)) |> 
                    x -> reshape(x, (2,:, size(rpn_class_logits, ndims(rpn_class_logits))))

  ss = []
  for i in 1:size(rpn_class_logits, ndims(rpn_class_logits))
    y = selectdim(rpn_class_logits, ndims(rpn_class_logits), i)
    push!(ss, softmax(y))
  end
  rpn_probs = cat(ss..., dims = 3)

  rpn_bbox = c.conv_bbox(x)
  rpn_bbox = permutedims(rpn_bbox, (3,4,2,1)) |> 
                    x -> reshape(x, (4,:, size(rpn_bbox, ndims(rpn_bbox))))

  rpn_class_logits, rpn_probs, rpn_bbox
end

mutable struct BottleNeck
  chain
  downsample
end

@treelike BottleNeck

function BottleNeck(inplanes::Int, planes::Int; stride = (1,1), downsample=nothing)
  chain = Chain(ConvBlock((1,1), inplanes=>planes,
                                stride = stride, 
                                pad = (0,0)),
                ConvBlock((3,3), planes=>planes, pad = (1,1)),
                Conv((1,1), planes=>4*planes),
                BatchNorm(4*planes)) |> gpu
  if downsample isa Nothing
    downsample = identity
  end
  BottleNeck(chain, downsample)
end

function (c::BottleNeck)(x)
  # @show "input size: $(size(x))"
  residual = x
  out = c.chain(x)
  if !isa(c.downsample, Nothing)
    residual = c.downsample(x)
  end
  out = out .+ residual
  relu.(out)
end

mutable struct ResNet
  C1
  C2
  C3
  C4
  C5
end

@treelike ResNet

function make_layer(block, planes, blocks, inplanes; stride = (1,1))
  downsample = nothing  
  if any([stride != (1,1), inplanes != planes*4])
    downsample = Chain(Conv((1,1), inplanes => 4*planes, stride = stride),
                      BatchNorm(4*planes)) |> gpu
  end

  layers = []
  push!(layers, block(inplanes, planes; stride = stride, downsample = downsample))
  inplanes = planes * 4
  for i = 1:(blocks-1)
    push!(layers, block(inplanes, planes))
  end

  Chain(layers...) |> gpu
end

function ResNet(architecture::String; stage5::Bool = false)
  inplanes = 64
  if architecture == "resnet50"
    layers = [3,4,6,3]
  else
    layers = [3,4,23,3]
  end

  C1 = Chain(ConvBlock((7,7), 3=>64, stride = (2,2), pad = (4 ,4)),
            MaxPool((3,3), stride = (2,2))) |> gpu


  C2 = make_layer(BottleNeck, 64, layers[1], inplanes)
  C3 = make_layer(BottleNeck, 128, layers[2], 256, stride = (2,2))
  C4 = make_layer(BottleNeck, 256, layers[3], 512, stride = (2,2))
  if stage5
    C5 = make_layer(BottleNeck, 512, layers[4], 1024, stride = (2,2))
  else
    C5 = identity
  end

  ResNet(C1, C2, C3, C4, C5)
end

function (c::ResNet)(x)
  x = c.C1(x)
  x = c.C2(x)
  x = c.C3(x)
  x = c.C4(x)
  x = c.C5(x)
  x
end

mutable struct MaskRCNN
  fpn
  anchors
  rpn
  classifier
  mask
end

@treelike MaskRCNN

function build(config = nothing)
  resnet = ResNet("resnet101", stage5=true)
  fpn = FPN(256,
            resnet.C1,
            resnet.C2,
            resnet.C3,
            resnet.C4,
            resnet.C5)

  RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]
  RPN_ANCHOR_RATIOS = [0.5, 1, 2]
  BACKBONE_SHAPES = [256 256;
                     128 128;
                      64  64;
                      32  32;
                      16  16;]
  BACKBONE_STRIDES = [4, 8, 16, 32, 64]
  RPN_ANCHOR_STRIDE = 1
  anchors = generate_pyramid_anchors(RPN_ANCHOR_SCALES,
                                    RPN_ANCHOR_RATIOS,
                                    BACKBONE_SHAPES,
                                    BACKBONE_STRIDES,
                                    RPN_ANCHOR_STRIDE)

  rpn = RPN(length(RPN_ANCHOR_RATIOS), (RPN_ANCHOR_STRIDE,RPN_ANCHOR_STRIDE), 256)
  POOL_SIZE = 7
  IMAGE_SHAPE = [1024, 1024, 3]
  NUM_CLASSES = 81
  classifier = Classifier(256, POOL_SIZE, IMAGE_SHAPE, NUM_CLASSES)

  mask = Mask(256, 2*POOL_SIZE, IMAGE_SHAPE, NUM_CLASSES)

  MaskRCNN(
    fpn,
    anchors,
    rpn,
    classifier,
    mask)

end


function detect(c::MaskRCNN, images)
  molded_images, image_metas, windows = mold_inputs(images)
  detections, mrcnn_mask = predict(molded_images, image_metas)
end

function predict(c::MaskRCNN, molded_images, image_metas,
                gt_class_ids = nothing, gt_boxes = nothing,
                gt_masks = nothing, mode = "inference")

  p2_out, p3_out, p4_out, p5_out, p6_out = c.fpn(molded_images);
  rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out];
  mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out];

  layer_outputs = []
  for p in rpn_feature_maps
      push!(layer_outputs, c.rpn(p))
  end

  ops = zip(layer_outputs...)
  ops2 = []
  for o in ops
    push!(ops2, reduce(hcat, o))
  end

  rpn_class_logits, rpn_class, rpn_bbox = ops2; # map(x -> reduce(hcat, x), ops)

  POST_NMS_ROIS_TRAINING = 2000
  POST_NMS_ROIS_INFERENCE = 1000
  RPN_NMS_THRESHOLD = 0.7f0

  if mode == "training"
    proposal_count = POST_NMS_ROIS_TRAINING
  else
    proposal_count = POST_NMS_ROIS_INFERENCE
  end

  rpn_rois_arr = []

  for i in 1:size(rpn_class, ndims(rpn_class))
    rpn_class_slice = @view rpn_class[:,:,i] 
    rpn_bbox_slice = @view rpn_bbox[:, :, i]

    rpn_rois = proposal_layer(rpn_class_slice',
                rpn_bbox_slice',
                proposal_count = proposal_count,
                nms_threshold = RPN_NMS_THRESHOLD,
                anchors = c.anchors)
    push!(rpn_rois_arr, rpn_rois)
  end

  # Make rpn_rois separate for every image. Currently, all images share
  # all the ROIs.
  rpn_rois = reduce(vcat, rpn_rois_arr)  # remove

  if mode == "inference"
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = c.classifier(mrcnn_feature_maps, rpn_rois)

    detections = detection_layer(rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

    IMAGE_SHAPE = (1024, 1024, 3)
    h, w = IMAGE_SHAPE[1:2]
    scale = [h w h w] |> gpu
    detection_boxes = detections[:, 1:4] ./ scale

    mrcnn_mask = c.mask(mrcnn_feature_maps, detection_boxes)

    return [detections, mrcnn_mask]
  elseif mode == "training"

    IMAGE_SHAPE = (1024, 1024, 3)
    h, w = IMAGE_SHAPE[1:2]
    scale = [h w h w] |> gpu

    gt_boxes = gt_boxes ./ scale;

    rois, target_class_ids, target_deltas, target_mask =
                detection_target_layer(rpn_rois, gt_class_ids,
                                      gt_boxes, gt_masks)

    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = c.classifier(mrcnn_feature_maps, rois)

    mrcnn_mask = c.mask(mrcnn_feature_maps, rois)

    return (rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits,
            target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
    # use o/p from here, add targets from datatset.jl
    # compute loss here and push backprop
  else
    error("Mode $mode not implemented")
  end
end

function train_maskrcnn(c::MaskRCNN, dataset = "coco"; epochs = 100, images_per_batch = 2)
  if dataset != "coco"
    return "Define $dataset API, or use the COCO API"
  end

  cid, images, classes = get_class_ids()
  c = build()
  masks = Dict()

  # ps_mask = params(c.mask.chain)
  # ps_classifier = params(c.classifier)
  # ps_fpn = params(c.fpn)
  # ps_rpn = params(c.rpn)
  # ps = Params([ps_fpn, ps_rpn, ps_classifier, ps_mask])
  ps = params(c)
  opt = Descent()

  total_images = length(images)
  for epoch in 1:epochs

    # img_data, mask, img_class, rpn_bbox, masks = sample_coco(cid, images, classes, masks = masks)
    image, image_metas, gt_class_ids, gt_boxes, gt_masks = load_image_gt(cid, images, classes)
    # molded_image, image_metas, windows = mold_inputs((img_data,))

    image = Float32.(reshape(image, (size(image)..., 1))) |> gpu;
    gt_masks = gpu(gt_masks);
    gt_boxes = gpu(gt_boxes);

    rpn_match, rpn_bbox = build_rpn_targets(size(image)[1:2], c.anchors, gt_class_ids, gt_boxes)

    rpn_class_logits, rpn_pred_bbox, target_class_ids,
    mrcnn_class_logits, target_deltas, mrcnn_bbox,
    target_mask, mrcnn_mask = predict(c,
                              image,
                              image_metas,
                              gt_class_ids,
                              gt_boxes,
                              gt_masks,
                              "training")

    target_class_ids = Int.(target_class_ids);

    gs = Tracker.gradient(ps) do
      rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
      rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
      mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
      mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
      mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

      loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
    end
    back!(loss)

    update!(opt, ps, gs)
  end
end

include("utils.jl")

