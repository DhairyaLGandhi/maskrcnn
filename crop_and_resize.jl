# using Flux, Flux.Tracker, Flux.Optimise
using Base.Threads

function crop_and_resize(images::NTuple, boxes, boxes_ind,
						crop_height, crop_width, extrapolation_value)
	out = []
	for i in images
		@show size(i)
		t = crop_and_resize(i, boxes, boxes_ind,
								crop_height, crop_width, extrapolation_value)
		@show size(t)
		push!(out, t)
	end
	# cat(out..., dims = 4)
	@show out[1] == out[2]
	out
end

function crop_and_resize(image, boxes, box_ind, 
						crop_height, crop_width,
						extrapolation_value=0.0f0)
	batch_size = size(image, 4)
	depth = size(image, 3)
	image_height = size(image, 2)
	image_width = size(image, 1)

	num_boxes = size(boxes, 1)
	# crops = param(zero(image))
	image_data = Tracker.data(image)
	boxes_data = Tracker.data(boxes)
	box_ind_data = Tracker.data(box_ind)

	@show crop_height, crop_width, depth, num_boxes
	crops = param(zeros(Float32, crop_height, crop_width, depth, num_boxes))
	crops_data = Tracker.data(crops)

	_crop_and_resize_per_box!(
		image_data,
		batch_size,
        depth,
        image_height,
        image_width,

        boxes_data,
        box_ind_data,
        1,
        num_boxes,

        crops_data,
        crop_height,
        crop_width,
        extrapolation_value)
	crops
end


function _crop_and_resize_per_box!(image_data, 
								batch_size,
								depth,
								image_height,
								image_width,

								boxes_data,
								boxes_index_data,
								start_box,
								limit_box,

								corps_data,
								crop_height,
								crop_width,
								extrapolation_value)

	image_channel_elements = image_height * image_width
	image_elements = depth * image_channel_elements
	channel_elements = crop_height * crop_width
	crop_elements = depth * channel_elements

	# @show limit_box
	@threads for b = start_box:limit_box
		box = boxes_data[b,:]
		y1 = box[1]
		x1 = box[2]
		y2 = box[3]
		x2 = box[4]
		
		b_in = boxes_index_data[b]
		if b_in <= 0 || b_in > batch_size
			error("Error: box index $(b_in), not in range of images [1,$(batch_size))")
		end

		height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1.0f0) / (crop_height - 1) : 0.0f0
		width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1.0f0) / (crop_width - 1) : 0.0f0

		for y = 1:crop_height
			in_y = (crop_height > 1) ?
				y1 * (image_height - 1) + y * height_scale :
				0.5f0 * (y1 + y2) * (image_height - 1)

			if in_y < 0 || in_y > (image_height - 1)
				for x = 1:crop_width
					for d = 1:depth
						corps_data[y,x,d,b] = extrapolation_value
					end
				end
				continue
			end

			top_y_index = floor(Int, in_y)
			top_y_index == 0 && (top_y_index = 1)
			bottom_y_index = ceil(Int, in_y)
			bottom_y_index == 0 && (bottom_y_index = crop_height)
			y_lerp = in_y - top_y_index

			for x = 1:crop_width
				in_x = (crop_width > 1) ?
                   x1 * (image_width - 1) + x * width_scale :
                   0.5f0 * (x1 + x2) * (image_width - 1)

                if in_x < 0 || in_x > image_width - 1

                	for d = 1:depth
                		corps_data[y,x,d,b] = extrapolation_value   # b => b_in
                	end
                	continue
                end

                left_x_index = floor(Int, in_x)
                left_x_index == 0 && (left_x_index = 1)
                right_x_index = ceil(Int, in_x)
                right_x_index == 0 && (right_x_index = crop_width)
                x_lerp = in_x - left_x_index

                # @show b, b_in
                for d = 1:depth
                	pimage = image_data[:,:,d,b_in]   # b => b_in

                	top_left = pimage[top_y_index, left_x_index]
                	top_right = pimage[top_y_index, right_x_index]
                	bottom_left = pimage[bottom_y_index, left_x_index]
                	bottom_right = pimage[bottom_y_index, right_x_index]

                	top = top_left + (top_right - top_left) * x_lerp
                	bottom = bottom_left + (bottom_right - bottom_left) * x_lerp

                	corps_data[y,x,d,b] = top + (bottom - top) * y_lerp    # b => b_in
                end
            end
        end
    end
    corps_data
end



function ∇crop_and_resize!(grads, boxes, boxes_index, grads_image)
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

	Tracker.zero_grad!(grads_image.grad)

	grads_data = Tracker.data(grads)
	boxes_data = Tracker.data(boxes)
	boxes_index_data = Tracker.data(boxes_index)
	grads_image_data = Tracker.data(grads_image)

	for b = 1:num_boxes
		box = boxes_data[b,:]

		y1 = box[1]
		x1 = box[2]
		y2 = box[3]
		x2 = box[4]

		b_in = boxes_index_data[b]
		if b_in <= 0 || b_in > batch_size
			error("Error, $(b_in) not in [1,$(batch_size))")
		end

		height_scale = crop_height > 1 ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0.0f0
		width_scale = crop_height > 1 ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0.0f0

		for y = 1:crop_height
			in_y = (crop_height > 1) ?
               y1 * (image_height - 1) + y * height_scale :
               0.5f0 * (y1 + y2) * (image_height - 1)

            if in_y < 0 || in_y > (image_height -1)
            	continue
            end

            top_y_index = floor(Int, in_y)
            bottom_y_index = ceil(Int, in_y)

            y_lerp = in_y - top_y_index

            for x = 1:crop_width
            	in_x = (crop_width > 1) ?
                   x1 * (image_width - 1) + x * width_scale :
                   0.5f0 * (x1 + x2) * (image_width - 1)

                if in_x < 0 || in_x > (image_width -1)
                	continue
                end

                left_x_index = floor(Int, in_x)
                right_x_index = ceil(Int, in_x)
                x_lerp = in_x - left_x_index
                for d = 1:depth
                	pimage = grads_image_data[:,:,d,b_in]

                	grad_val = grads_data[x,y,d,b]
                	
                	dtop = (1 - y_lerp) * grad_val
                	grads_image_data[top_y_index, left_x_index,d,b_in] += (1 - x_lerp) * dtop
                	grads_image_data[top_y_index, right_x_index,d,b_in] += x_lerp * dtop

                	dbottom = y_lerp * grad_val
                	grads_image_data[bottom_y_index, left_x_index,d,b_in] += (1 - x_lerp) * dbottom
                	grads_image_data[bottom_y_index, right_x_index,d,b_in] += x_lerp * dbottom
                end
            end
        end
    end
    grads
end

@grad function crop_and_resize(image, boxes, box_ind,
                       crop_height, crop_width,
                       extrapolation_value)

    y = crop_and_resize(Tracker.data(image),
            Tracker.data(boxes),
            Tracker.data(box_ind),
            crop_height, crop_width,
            extrapolation_value)

    y, Δ -> (Tracker.nobacksies(:crop_and_resize,
    	∇crop_and_resize!(Tracker.data(Δ),
            Tracker.data(boxes),
            Tracker.data(boxes_index),
            Tracker.data(y))), nothing, nothing, nothing, nothing, nothing)
end

∇crop_and_resize(grads::AbstractArray, boxes::TrackedArray,
				boxes_index::TrackedArray, grads_image::TrackedArray) = 
		Tracker.track(∇crop_and_resize, grads, boxes, boxes_index, grads_image)