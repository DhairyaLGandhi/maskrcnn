# using Flux, Flux.Tracker, Flux.Optimise
using Base.Threads
using BSON: @save, @load

function crop_and_resize(image, boxes, box_ind;
                        crop_height = 28, crop_width = 28,
			extrapolation_value = 0.f0)

    @show crop_height
    batch_size = size(image, 4)
    depth = size(image, 3)
    image_height = size(image, 2)
    image_width = size(image, 1)
    num_boxes = size(boxes, 1)
    crop_height = round.(Int, crop_height)
    crop_width = round.(Int, crop_width)

    # @show crop_width
    crops = similar(image, crop_height, crop_width, depth, num_boxes)

    boxes_data = boxes
    boxes_data = transpose(boxes_data)
    bs = 1:batch_size
    image_data = image
    @assert crop_height > 1
    @assert crop_width > 1
    for (b, box) in enumerate(eachcol(boxes_data))
        y1, x1, y2, x2 = box

        b_in = Int.(box_ind[b])
        # @show b_in
        b_in in bs || error("Error: box index $(b_in), not in range of images [1,$(batch_size))")

        height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1.0f0) / (crop_height - 1.0f0) : 0.0f0
        width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1.0f0) / (crop_width - 1.0f0) : 0.0f0
        for p = CartesianIndices((1:crop_width, 1:crop_height))
            y, x = p.I
            in_y = y1 * (image_height - 1) + y * height_scale
            if in_y < 0 || in_y > image_width
                crops[y, 1:crop_width, 1:depth, b] .= extrapolation_value
                continue
            end

            top_y_index = floor.(Int, in_y)
            top_y_index == 0 && (top_y_index = 1)
            bottom_y_index = ceil.(Int, in_y)
            bottom_y_index == 0 && (bottom_y_index = crop_height)
            y_lerp = in_y - top_y_index

            in_x = x1 * (image_width - 1) + x * width_scale
            if in_x < 0 || in_x > image_width
                crops[y,x, 1:depth, b] .= extrapolation_value
                continue
            end

            left_x_index = floor.(Int, in_x)
            left_x_index == 0 && (left_x_index = 1)
            right_x_index = ceil.(Int, in_x)
            right_x_index == 0 && (right_x_index = crop_width)
            x_lerp = in_x - left_x_index



            top_left = @views image_data[top_y_index, left_x_index, :, b_in]
            top_right = @views image_data[top_y_index, right_x_index, :, b_in]
            bottom_left = @views image_data[bottom_y_index, left_x_index, :, b_in]
            bottom_right = @views image_data[bottom_y_index, right_x_index, :, b_in]

            top = @. top_left + (top_right - top_left) * x_lerp
            bottom = @. bottom_left + (bottom_right - bottom_left) * x_lerp
            val = @. top + (bottom - top) * y_lerp
            # @show typeof(val)
            crops[y,x,:,b] .= val
        end
    end
    @show size(crops)
    round.(crops)

end


function ∇crop_and_resize!(grads, boxes, boxes_index, grads_image; kw...)
	@show "in  ∇crop_and_resize!"
	batch_size = size(grads_image, ndims(grads_image))
	depth = size(grads_image, 3)
	image_height = size(grads_image, 2)
	image_width = size(grads_image, 1)

	num_boxes = size(grads, ndims(grads))
	crop_height = size(grads, 2)
	crop_width = size(grads, 1)

	image_channel_elements = image_height * image_width
	image_elements = depth * image_channel_elements

	channel_elements = crop_height * crop_width
	crop_elements = depth * channel_elements

	# grads_image = Tracker.zero(grads_image)

	grads_data = grads # Tracker.data(grads)
	boxes_data = boxes # Tracker.data(boxes)
	# boxes_data = boxes
	boxes_index_data = boxes_index # Tracker.data(boxes_index)
	grads_image_data = zero(grads_image)
	bs = 1:batch_size

	@assert crop_height > 1
    @assert crop_width > 1
	for (b, box) in enumerate(eachcol(boxes_data))
		y1, x1, y2, x2 = box

		b_in = Int.(Tracker.data(boxes_index[b]))
        b_in in bs || error("Error: box index $(b_in), not in range of images [1,$(batch_size))")

        height_scale = crop_height > 1 ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0.0f0
		width_scale = crop_height > 1 ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0.0f0

		for p = CartesianIndices((1:crop_width, 1:crop_height))
            y, x = p.I

            in_y = (crop_height > 1) ?
			   y1 * (image_height - 1) + y * height_scale :
			   0.5f0 * (y1 + y2) * (image_height - 1)

			if in_y < 0 || in_y > image_height
				continue
			end

			# top_y_index = floor.(Int, in_y)
			# bottom_y_index = ceil.(Int, in_y)

			top_y_index = floor.(Int, in_y)
            top_y_index == 0 && (top_y_index = 1)
            bottom_y_index = ceil.(Int, in_y)
            bottom_y_index == 0 && (bottom_y_index = crop_height)

			y_lerp = in_y - top_y_index

			in_x = (crop_width > 1) ?
				   x1 * (image_width - 1) + x * width_scale :
				   0.5f0 * (x1 + x2) * (image_width - 1)

			if in_x < 0 || in_x > (image_width -1)
				continue
			end

			# left_x_index = floor.(Int, in_x)
			# right_x_index = ceil.(Int, in_x)
			left_x_index = floor.(Int, in_x)
            left_x_index == 0 && (left_x_index = 1)
            right_x_index = ceil.(Int, in_x)
            right_x_index == 0 && (right_x_index = crop_width)

			x_lerp = in_x - left_x_index

			grad_val = grads_data[x,y,:,b]

			ty = copy(top_y_index) |> Tracker.data |> Int
			by = copy(bottom_y_index) |> Tracker.data |> Int
			lx = copy(left_x_index) |> Tracker.data |> Int
			rx = copy(right_x_index) |> Tracker.data |> Int

			# top_y_index = Int.(Tracker.data(top_y_index))
			# bottom_y_index = Int.(Tracker.data(bottom_y_index))
			# left_x_index = Int.(Tracker.data(left_x_index))
			# right_x_index = Int.(Tracker.data(right_x_index))

			dtop = @. (1 - y_lerp) * grad_val
			# @show (1.f0 .- x_lerp) .* dtop
			p1 = Tracker.data((1.f0 .- x_lerp) .* dtop)
			p2 = Tracker.data(x_lerp .* dtop)
			grads_image_data[ty,lx,:,b_in] += p1 # (1.f0 .- x_lerp) .* dtop
			grads_image_data[ty,rx,:,b_in] += p2 # x_lerp .* dtop

			dbottom = @. y_lerp * grad_val
			p3 = Tracker.data((1.f0 .- x_lerp) .* dbottom)
			p4 = Tracker.data(x_lerp .* dbottom)
			grads_image_data[by, lx,:,b_in] += p3 # (1.f0 .- x_lerp) .* dbottom
			grads_image_data[by, rx,:,b_in] += p4 # x_lerp .* dbottom
		end
	end
	@show "done"
	@show sum(grads_image_data)
	grads_image_data
end

@grad function crop_and_resize(image::AbstractArray,
			boxes::AbstractArray,
			box_ind::AbstractArray;
			kw...)

    @info "doing crop_and_resize backprop"
    y = crop_and_resize(Tracker.data(image),
                    Tracker.data(boxes),
                    Tracker.data(box_ind);
                    kw...)

    y, Δ -> (∇crop_and_resize!(Δ,
                    boxes,
                    box_ind,
                    image; kw...), boxes, 0.f0)
end


crop_and_resize(x::TrackedArray,
		y::Union{TrackedArray, AbstractArray},
		z::Union{TrackedArray, AbstractArray};
		kw...) = Tracker.track(crop_and_resize, x, y, z; kw...)

crop_and_resize(x::AbstractArray,
		y::TrackedArray,
		z::Union{TrackedArray, AbstractArray};
		kw...) = Tracker.track(crop_and_resize, x, y, z; kw...)

crop_and_resize(x::AbstractArray,
		y::AbstractArray,
		z::TrackedArray;
		kw...) = Tracker.track(crop_and_resize, x, y, z; kw...)

crop_and_resize(x::TrackedArray,
		y::TrackedArray,
		z::TrackedArray;
		kw...) = Tracker.track(crop_and_resize, x, y, z; kw...)

crop_and_resize(x::TrackedArray,
		y::TrackedArray,
		z::AbstractArray;
		kw...) = Tracker.track(crop_and_resize, x, y, z; kw...)
