// Auto-generated PDAQP configuration
// Source: pdaqp.h, pdaqp.c

`define PDAQP_N_PARAMETER 10
`define PDAQP_N_SOLUTION 5
`define PDAQP_TREE_NODES 131
`define PDAQP_HALFPLANES 352
`define PDAQP_FEEDBACKS 880

`define PDAQP_SOL_PER_NODE 55
`define PDAQP_HALFPLANE_STRIDE 11
`define PDAQP_ESTIMATED_BST_DEPTH 10

// Fixed-point Q format for halfplanes
`define HALFPLANE_INT_BITS 2
`define HALFPLANE_FRAC_BITS 14
`define HALFPLANE_SCALE_FACTOR 16384

// Fixed-point Q format for feedbacks
`define FEEDBACK_INT_BITS 2
`define FEEDBACK_FRAC_BITS 14
`define FEEDBACK_SCALE_FACTOR 16384

`define INPUT_DATA_WIDTH 16
`define OUTPUT_DATA_WIDTH 16

// Backward compatibility
`define PDAQP_FIXED_POINT_BITS 14
`define PDAQP_SCALE_FACTOR 16384
`define INPUT_INT_BITS 2
`define OUTPUT_INT_BITS 2
`define OUTPUT_FRAC_BITS 14
