using JSON, StatsBase
using Images, Luxor
using Images.ImageTransformations
using Interpolations
# using ImageView

const base = "/home/shared/coco_images"
const images_path = joinpath(base, "train2014")
const annotations_path = joinpath(base, "annotations")
const masks_path = joinpath(base, "masks")

function coco_annotations(annotations_path = annotations_path)
  anns = filter(x -> occursin("json",x), readdir(annotations_path))
  n = Dict()

  for i in anns
    s = String(JSON.read(joinpath(annotations_path, i)))
    s = JSON.parse(s)
    n[i] = s
  end

  n
end

# COCO_train2014_000000581882.jpg
# 480023

coco = coco_annotations()

function get_class_ids()
  class_ids = Integer[]
  class_ids_dic = Dict()
  images = Dict()
  for v in coco["instances_train2014.json"]["annotations"]
    push!(class_ids, v["category_id"])
    if !haskey(class_ids_dic, v["category_id"])
      class_ids_dic[v["category_id"]] = [v["image_id"]]
    else
      class_ids_dic[v["category_id"]] = push!(class_ids_dic[v["category_id"]], v["image_id"])
    end

    if !haskey(images, v["image_id"])
      images[v["image_id"]] = Dict()
      images[v["image_id"]]["vals"] = [v]
    else
      push!(images[v["image_id"]]["vals"], v)
    end
  end

  for v in coco["instances_train2014.json"]["images"]
    try
      images[v["id"]]["file_name"] = v["file_name"]
    catch ex
    end
  end
  class_ids_dic, images, sort(unique(class_ids))
end

function sample_coco(cid, images, classes;
                    base_name = "COCO_train2014_000000", masks = Dict())

  img_class = 4 # sample(classes) # 4
  img_id = 275544 # sample(cid[img_class]) # 275544
  img = images[img_id]
  # segmentation = img["segmentation"]
  # rpn_bbox = transpose(img["bbox"])
  img_data = try
      Images.load(joinpath(images_path, img["file_name"]))
    catch KeyError
      Images.load(images_path * "/$base_name" * string(img_id) * ".jpg")
    end
  img_data = permutedims(channelview(img_data), (3,2,1))
  # mask = zeros(Float32, size(img_data)[1:2]...)
  mask, bboxes, class_ids, masks = make_masks(img, size(img_data)[1:2], img["file_name"], masks = masks)
  
  # Don't resize
  # img_data = imresize(img_data, 299,299)
  img_data, mask, class_ids, bboxes, img_id, masks
end

function sample_coco(id::Int, images; base_name = "COCO_train2014_000000", masks = Dict())
  img = images[id]
  if haskey(img, "file_name")
    img_data = Images.load(joinpath(images_path, img["file_name"]))
  else
    try
      img_data = Images.load(images_path * "/$base_name" * string(id) * ".jpg")
    catch ex
      @show ex
      throw(ex)
    end
  end
  img_data = permutedims(channelview(img_data), (3,2,1))
  # img_class = img["category_id"]
  # segmentation = img["segmentation"]
  # mask = zeros(Float32, size(img_data)[1:2]...)
  # mask, masks = make_masks(segmentation, size(img_data)[1:2], img["file_name"], masks = masks)
  mask, bboxes, class_ids, masks = make_masks(img, size(img_data)[1:2], img["file_name"], masks = masks)

  # rpn_bbox = transpose(img["bbox"])
  img_data, mask, class_ids, bboxes, id, masks
end

function make_masks(segmentation::Array, image_size::Tuple, image_name; masks::Dict = Dict())
  polyX = segmentation[1][1:2:end]
  polyY = segmentation[1][2:2:end]

  if !haskey(masks, image_name)
    image_name = split(image_name, ".")[1] * ".png"
    # @png begin
    #   Drawing(image_size..., joinpath(base, "masks/mask_" * image_name))
    #   ps = map(x -> Luxor.Point(x[2],x[1]), zip(polyY, polyX))
    #   poly(ps, :fill)

    # end

    Drawing(image_size..., joinpath(masks_path, "mask_" * image_name))
    background("black")
    sethue("white")
    ps = map(x -> Luxor.Point(x[2],x[1]), zip(polyY, polyX))
    poly(ps, :fill)
    finish()

    masks[image_name] = joinpath(masks_path, "mask_"*image_name)
    i = Images.load(joinpath(masks_path, "mask_"*image_name))
    i = permutedims(channelview(i), (3,2,1))
    i = sum(i, dims = 3)
    i, masks
  else
    i = Images.load(joinpath(masks_path, "mask_"*image_name))
    i = colorview(Gray, i)
    i = permutedims(channelview(i), (3,2,1))
    i = sum(i, dims = 3)
    i, masks
  end
end

function make_masks(img::Dict, image_size::Tuple, image_name; masks = Dict())
  img_masks = []
  bboxes = []
  class_ids = []
  for i in img["vals"]
    bb = transpose(i["bbox"])
    bb = [bb[1] bb[2] bb[1] + bb[3] bb[2] + bb[4]]
    mask, masks = make_masks(i["segmentation"], image_size, image_name)
    push!(bboxes, bb)
    push!(img_masks, mask)
    if i["iscrowd"] == 0
      push!(class_ids, i["category_id"])
    else
      push!(class_ids, -1)
    end
  end

  cat(img_masks..., dims = 3), reduce(vcat, bboxes), class_ids, masks
end


function resize_mask(mask, scale, padding)
  h, w = size(mask)[1:2]
  z = zeros(round(Int, scale*w), round(Int, scale*h), size(mask, ndims(mask)))
  s = BSpline(Cubic(Flat(OnCell())))
  itp = interpolate(mask, (s, s, NoInterp()))
  ImageTransformations.imresize!(z, itp)
  z = round.(z)
  z
end

function minimise_mask(bbox, mask, mini_shape)
  s = (mini_shape..., size(mask, ndims(mask)))
  mini_masks = zeros(s...)
  # @show size(mask)
  for (i, box) in zip(1:size(mask, ndims(mask)), eachrow(bbox))
    # @show box
    m = mask[:, :, i]
    y1, x1, y2, x2 = box
    m = m[x1:x2, y1:y2]
    # @show size(m)
    # @show "here"
    m = imresize(m, mini_shape)
    # @show size(m)
    m = clamp.(m, 0., 1.)
    mini_masks[:,:,i] = m
  end
  # mini_masks = round.(masks)
  mini_masks
end

function load_image_gt(cid::Dict, images::Dict, classes; augment = true, use_mini_mask = true)
  img_data, mask, class_ids, rpn_bbox, image_id, masks = sample_coco(cid, images, classes)
  resized_img, window, scale, padding = resize_image(img_data, 1024, 1024, true)
  mask = resize_mask(mask, scale, padding)
  # return mask

  # if augment && rand() > 0.5
  #   resized_img = imshow(resized_img, flipx = true)
  #   mask = imshow(mask, flipx = true)
  # end

  bbox = extract_bboxes(mask)

  active_classes = zeros(length(classes))
  active_classes .= 1

  MINI_MASK_SHAPE = (56, 56)
  if use_mini_mask
    mask = minimise_mask(bbox, mask, MINI_MASK_SHAPE)
    # mask = minimise_mask(clamp.(round.(Int, rpn_bbox), 1, size(img_data, 2)), mask, MINI_MASK_SHAPE)
  end

  image_meta = compose_image_meta(image_id, size(img_data), window, active_classes)

  resized_img, image_meta, class_ids, bbox, mask
end


function load_image_gt(image_name::String, segmentation::Array, classes::Array; use_mini_mask = true)
  # mask_path = joinpath(masks_path, "mask_" * split(image_name, ".")[1] * ".png")

  img_data = load(image_name)
  i, masks = make_masks(segmentation, size(img_data)[1:2], image_name)

  img_data = permutedims(channelview(img_data), (3,2,1))
  resized_img, window, scale, padding = resize_image(img_data, 1024, 1024, true)
  i = resize_mask(i, scale, padding)

  bbox = extract_bboxes(i)

  active_classes = zeros(length(classes))
  active_classes .= 1

  MINI_MASK_SHAPE = (56, 56)
  if use_mini_mask
    mask = minimise_mask(bbox, i, MINI_MASK_SHAPE)
    # mask = minimise_mask(clamp.(round.(Int, rpn_bbox), 1, size(img_data, 2)), mask, MINI_MASK_SHAPE)
  end

  image_meta = compose_image_meta(667, size(img_data), window, active_classes)

  resized_img, image_meta, classes, bbox, mask
end
