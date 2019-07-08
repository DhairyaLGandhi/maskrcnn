using HDF5
import Base.show_unquoted

global arr = []
global ks = readlines("weights/maskrcnn_keys.txt");

function load_weights(c::MaskRCNN, hdfile = "weights/maskrcnn.h5")
	hd = h5open(hdfile, "r")
	mnets = [:fpn, :rpn, :classifier, :mask]

	d = Dict("fpn" => Symbol.(["C1", "C2", "C3", "C4", "C5", "P2_conv1", "P2_conv2",
						"P3_conv2", "P4_conv2", "P5_conv2", "P3_conv1",
						"P4_conv1", "P5_conv1", "P6"]),

			"rpn" => Symbol.(["conv_bbox", "conv_class", "conv_shared"]),
			"classifier" => Symbol.(["chain", "linear_bbox", "linear_class"]),
			"mask" => Symbol.(["chain"]))

	fs = []
	for k in mnets
		q = quote
			$k = filter(x -> occursin($(sprint(show_unquoted, k)), x), ks)
		end
		push!(fs, q)
	end

	eval.(fs);

	fs = []
	for k in d["fpn"]
		q = quote
			$k = filter(x -> occursin($(sprint(show_unquoted, k)), x), fpn)
		end
		push!(fs, q)
	end

	for k in d["rpn"]
		q = quote
			$k = filter(x -> occursin($(sprint(show_unquoted, k)), x), rpn)
		end
		push!(fs, q)
	end

	for k in d["classifier"]
		q = quote
			$k = filter(x -> occursin($(sprint(show_unquoted, k)), x), classifier)
		end
		push!(fs, q)
	end

	# Don't do it here otherwise `chain` will get rewritten
	# for k in d["mask"]
	# 	q = quote
	# 		$k = filter(x -> occursin($(sprint(show_unquoted, k)), x), mask)
	# 	end
	# 	push!(fs, q)
	# end

	eval.(fs);

	for f in fieldnames(typeof(c))
        v = getfield(c, f)
        for g in fieldnames(typeof(v))
        	if f == :fpn
	        	t = getfield(v, g)
	        	if g == :C1
	        		a = ["fpn.C1.0.weight",
	        			"fpn.C1.0.bias",
	        			"fpn.C1.1.bias",
	        			"fpn.C1.1.weight"]

	        		ws = [read(hd, x) for x in a]
	        		Flux.loadparams!(t, ws)
	        		continue
	        	end

	        	if occursin("C", string(g))
		            a = get_C_arr(t, Symbol(f,".", g))
			        ws = [read(hd, x) for x in a]
			        Flux.loadparams!(t, ws)

			        continue
			    else
			    	g == :P6 && continue
			    	a = eval(g)
			    	a = reverse(a)
			    	ws = [read(hd, x) for x in a]
			    	Flux.loadparams!(t, ws)
			    	continue
			    end
		    elseif f == :rpn
		    	t = getfield(v, g)
		    	a = eval(g) |> reverse
		    	ws = [read(hd, x) for x in a]
		    	Flux.loadparams!(t, ws)
		    elseif f == :classifier
		    	t = getfield(v, g)
		    	if g == :chain
		    		a = ["classifier.conv1.weight",
		    			"classifier.conv1.bias",
		    			"classifier.bn1.bias",
		    			"classifier.bn1.weight",
		    			"classifier.conv2.weight",
		    			"classifier.conv2.bias",
		    			"classifier.bn2.bias",
		    			"classifier.bn2.weight"]

		    		ws = [read(hd, x) for x in a]
		    		Flux.loadparams!(t, ws)
		    		continue
		    	end
		    	g in d["classifier"] || continue
		    	a = eval(g) |> reverse
		    	ws = [occursin("weight", x) ? transpose(read(hd, x)) : read(hd, x) for x in a]
		    	Flux.loadparams!(t, ws)
		    elseif f == :mask
		    	t = getfield(v, g)
		    	a = get_mask_arr(t, "mask")
		    	ws = [read(hd, x) for x in a]
		    	Flux.loadparams!(t, ws)
		    	continue
		    end

        end
    end
    c
end

function get_C_arr(model, base = "fpn.C2")
	arr = []
	ds = true
	for i = 0:(length(model)-1), j = 1:3
		push!(arr, "$base." * string(i) * ".conv" * string(j) * ".weight")
		push!(arr, "$base." * string(i) * ".conv" * string(j) * ".bias")
		push!(arr, "$base." * string(i) * ".bn" * string(j) * ".bias")
		push!(arr, "$base." * string(i) * ".bn" * string(j) * ".weight")
		if j ==3 && ds
			push!(arr, "$base." * "0" * ".downsample" * ".0" * ".weight")
			push!(arr, "$base." * "0" * ".downsample" * ".0" * ".bias")
			push!(arr, "$base." * "0" * ".downsample" * ".1" * ".bias")
			push!(arr, "$base." * "0" * ".downsample" * ".1" * ".weight")
			ds = false
		end
	end
	arr
end

function get_mask_arr(model, base = "mask")
	arr = []

	for i = 1:(length(model)-1)

		if i == 5
			push!(arr, base * ".deconv" * ".weight")
			push!(arr, base * ".deconv" * ".bias")
			push!(arr, base * ".conv$i" * ".weight")
			push!(arr, base * ".conv$i" * ".bias")
			break
		end

		push!(arr, base * ".conv$i" * ".weight")
		push!(arr, base * ".conv$i" * ".bias")
		push!(arr, base * ".bn$i" * ".bias")
		push!(arr, base * ".bn$i" * ".weight")

	end
	arr
end
