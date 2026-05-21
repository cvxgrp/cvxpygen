// Auto-generated PDAQP configuration
// Source: pdaqp.h, pdaqp.c
// Mode: Fixed-Point

`define PDAQP_N_PARAMETER 2
`define PDAQP_N_SOLUTION 3
`define PDAQP_TREE_NODES 5
`define PDAQP_HALFPLANES 6
`define PDAQP_FEEDBACKS 27

`define PDAQP_SOL_PER_NODE 9
`define PDAQP_HALFPLANE_STRIDE 3
`define PDAQP_ESTIMATED_BST_DEPTH 4

// Data format selection
`define PDAQP_USE_FP32 0
`define PDAQP_USE_FP16 0
`define PDAQP_USE_FIX16 1
`define INPUT_DATA_WIDTH 16
`define OUTPUT_DATA_WIDTH 16

// Fixed-point mode (16-bit)
`define PDAQP_FIX16_MODE 1

// Fixed-point Q format for halfplanes
`define HALFPLANE_INT_BITS 2
`define HALFPLANE_FRAC_BITS 14
`define HALFPLANE_SCALE_FACTOR 16384

// Fixed-point Q format for feedbacks
`define FEEDBACK_INT_BITS 1
`define FEEDBACK_FRAC_BITS 15
`define FEEDBACK_SCALE_FACTOR 32768

// Backward compatibility
`define PDAQP_FIXED_POINT_BITS 15
`define PDAQP_SCALE_FACTOR 32768
`define INPUT_INT_BITS 1
`define OUTPUT_INT_BITS 1
`define OUTPUT_FRAC_BITS 15

