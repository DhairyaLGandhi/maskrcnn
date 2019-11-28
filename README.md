## MaskRCNN.jl

Working implementation of RCNN in Flux.jl

DONE: 
 - loss functions
 - data generator
 - coco pipelining
 - backprop
 - fix incorrect gradients
 - check where gradients are dropped

TODO:
 - fix infs in log (`box_refinement`)
 - Documentation
 - clean up the mess
 - GPU NMS
