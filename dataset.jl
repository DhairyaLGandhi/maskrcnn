using JSON, StatsBase
using Images, Luxor

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
  class_ids = []
  class_ids_dic = Dict()
  images = Dict()
  for v in coco["instances_train2014.json"]["annotations"]
    push!(class_ids, v["category_id"])
    if !haskey(class_ids_dic, v["category_id"])
      class_ids_dic[v["category_id"]] = [v["image_id"]]
    else
      class_ids_dic[v["category_id"]] = push!(class_ids_dic[v["category_id"]], v["image_id"])
    end
    images[v["image_id"]] = v
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

  img_class = sample(classes)
  img_id = sample(cid[img_class])
  img = images[img_id]
  segmentation = img["segmentation"]
  rpn_bbox = transpose(img["bbox"])
  img_data = try
      Images.load(joinpath(images_path, img["file_name"]))
    catch KeyError
      Images.load(images_path * "/$base_name" * string(img_id) * ".jpg")
    end
  img_data = permutedims(channelview(img_data), (3,2,1))
  mask = zeros(Float32, size(img_data)[1:2]...)
  
  # Don't resize
  # img_data = imresize(img_data, 299,299)
  img_data, mask, img_class, rpn_bbox
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
  img_class = img["category_id"]
  segmentation = img["segmentation"]
  # mask = zeros(Float32, size(img_data)[1:2]...)
  mask, masks = make_masks(segmentation, size(img_data)[1:2], img["file_name"], masks = masks)
  rpn_bbox = transpose(img["bbox"])
  img_data, mask, img_class, rpn_bbox, segmentation, masks
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
    i, masks
  else
    Images.load(joinpath(masks_path, "mask_"*image_name))
    i, masks
  end
end




