`timescale 1ns/1ps
`include "include/pdaqp_config.vh"

module pdaqp_bst_lut (
    input                               clk,
    input                               rst_n,
    input  signed [15:0]  param_in_0,
    input  signed [15:0]  param_in_1,
    input  signed [15:0]  param_in_2,
    input  signed [15:0]  param_in_3,
    input  signed [15:0]  param_in_4,
    input  signed [15:0]  param_in_5,
    input  signed [15:0]  param_in_6,
    input  signed [15:0]  param_in_7,
    input  signed [15:0]  param_in_8,
    input  signed [15:0]  param_in_9,
    input                               valid_in,
    output reg signed [15:0] sol_out_0,
    output reg signed [15:0] sol_out_1,
    output reg signed [15:0] sol_out_2,
    output reg signed [15:0] sol_out_3,
    output reg signed [15:0] sol_out_4,
    output reg                          valid_out
);

    // Distributed ROM for FPGA
    (* rom_style = "distributed" *) reg [15:0] halfplanes [0:`PDAQP_HALFPLANES-1];
    (* rom_style = "distributed" *) reg [15:0] feedbacks [0:`PDAQP_FEEDBACKS-1];
    (* rom_style = "distributed" *) reg [7:0] hp_list [0:`PDAQP_TREE_NODES-1];
    (* rom_style = "distributed" *) reg [7:0] jump_list [0:`PDAQP_TREE_NODES-1];

    initial begin
        $readmemh("include/pdaqp_halfplanes.mem", halfplanes);
        $readmemh("include/pdaqp_feedbacks.mem", feedbacks);
        $readmemh("include/pdaqp_hp_list.mem", hp_list);
        $readmemh("include/pdaqp_jump_list.mem", jump_list);
    end
    
    localparam MAX_BST_DEPTH = `PDAQP_ESTIMATED_BST_DEPTH;
    localparam PIPE_DEPTH = MAX_BST_DEPTH + 5;
    
    // Pipeline registers
    reg valid_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param0_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param1_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param2_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param3_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param4_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param5_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param6_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param7_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param8_pipe[0:PIPE_DEPTH-1];
    reg [15:0] param9_pipe[0:PIPE_DEPTH-1];
    
    // BST traversal (size = MAX_BST_DEPTH + 1 for [0:MAX_BST_DEPTH])
    reg [7:0] current_id_pipe[0:MAX_BST_DEPTH];
    reg [7:0] next_id_pipe[0:MAX_BST_DEPTH];
    reg [7:0] hp_idx_pipe[0:MAX_BST_DEPTH];
    reg traversal_done_pipe[0:MAX_BST_DEPTH];
    
    // Solution pipeline
    reg [15:0] feedback_base_pipe[0:1];
    reg signed [31:0] sol_temp_0[0:1];
    reg signed [31:0] sol_temp_1[0:1];
    reg signed [31:0] sol_temp_2[0:1];
    reg signed [31:0] sol_temp_3[0:1];
    reg signed [31:0] sol_temp_4[0:1];
    
    // Combinational temporaries
    reg signed [31:0] hp_val;
    reg signed [31:0] hp_thresh;
    reg decision;
    reg [7:0] new_id;
    
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < PIPE_DEPTH; i = i + 1) begin
                valid_pipe[i] <= 0;
                param0_pipe[i] <= 0;
                param1_pipe[i] <= 0;
                param2_pipe[i] <= 0;
                param3_pipe[i] <= 0;
                param4_pipe[i] <= 0;
                param5_pipe[i] <= 0;
                param6_pipe[i] <= 0;
                param7_pipe[i] <= 0;
                param8_pipe[i] <= 0;
                param9_pipe[i] <= 0;
            end
            
            for (i = 0; i <= MAX_BST_DEPTH; i = i + 1) begin
                current_id_pipe[i] <= 0;
                next_id_pipe[i] <= 0;
                hp_idx_pipe[i] <= 0;
                traversal_done_pipe[i] <= 0;
            end
            
            for (j = 0; j < 2; j = j + 1) begin
                feedback_base_pipe[j] <= 0;
                sol_temp_0[j] <= 0;
                sol_temp_1[j] <= 0;
                sol_temp_2[j] <= 0;
                sol_temp_3[j] <= 0;
                sol_temp_4[j] <= 0;
            end
            
            valid_out <= 0;
            sol_out_0 <= 0;
            sol_out_1 <= 0;
            sol_out_2 <= 0;
            sol_out_3 <= 0;
            sol_out_4 <= 0;
            
        end else begin
            
            // Stage 0: Input
            valid_pipe[0] <= valid_in;
            
            if (valid_in) begin
                param0_pipe[0] <= param_in_0;
                param1_pipe[0] <= param_in_1;
                param2_pipe[0] <= param_in_2;
                param3_pipe[0] <= param_in_3;
                param4_pipe[0] <= param_in_4;
                param5_pipe[0] <= param_in_5;
                param6_pipe[0] <= param_in_6;
                param7_pipe[0] <= param_in_7;
                param8_pipe[0] <= param_in_8;
                param9_pipe[0] <= param_in_9;
                
                current_id_pipe[0] <= 0;
                next_id_pipe[0] <= jump_list[0];
                hp_idx_pipe[0] <= hp_list[0];
                traversal_done_pipe[0] <= (jump_list[0] == 0);
            end
            
            // Stages 1 to MAX_BST_DEPTH: BST traversal
            for (i = 0; i < MAX_BST_DEPTH; i = i + 1) begin
                valid_pipe[i+1] <= valid_pipe[i];
                param0_pipe[i+1] <= param0_pipe[i];
                param1_pipe[i+1] <= param1_pipe[i];
                param2_pipe[i+1] <= param2_pipe[i];
                param3_pipe[i+1] <= param3_pipe[i];
                param4_pipe[i+1] <= param4_pipe[i];
                param5_pipe[i+1] <= param5_pipe[i];
                param6_pipe[i+1] <= param6_pipe[i];
                param7_pipe[i+1] <= param7_pipe[i];
                param8_pipe[i+1] <= param8_pipe[i];
                param9_pipe[i+1] <= param9_pipe[i];
                
                if (valid_pipe[i]) begin
                    if (traversal_done_pipe[i]) begin
                        current_id_pipe[i+1] <= current_id_pipe[i];
                        next_id_pipe[i+1] <= next_id_pipe[i];
                        hp_idx_pipe[i+1] <= hp_idx_pipe[i];
                        traversal_done_pipe[i+1] <= 1;
                    end else begin
                        hp_val = ($signed(param0_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+0])) + ($signed(param1_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+1])) + ($signed(param2_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+2])) + ($signed(param3_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+3])) + ($signed(param4_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+4])) + ($signed(param5_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+5])) + ($signed(param6_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+6])) + ($signed(param7_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+7])) + ($signed(param8_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+8])) + ($signed(param9_pipe[i]) * $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+9]));
                        hp_thresh = $signed(halfplanes[hp_idx_pipe[i]*`PDAQP_HALFPLANE_STRIDE+`PDAQP_N_PARAMETER]) << `HALFPLANE_FRAC_BITS;
                        decision = hp_val <= hp_thresh;
                        new_id = next_id_pipe[i] + (decision ? 8'd1 : 8'd0);
                        
                        current_id_pipe[i+1] <= new_id;
                        next_id_pipe[i+1] <= new_id + jump_list[new_id];
                        hp_idx_pipe[i+1] <= hp_list[new_id];
                        traversal_done_pipe[i+1] <= (jump_list[new_id] == 0);
                    end
                end else begin
                    current_id_pipe[i+1] <= 0;
                    next_id_pipe[i+1] <= 0;
                    hp_idx_pipe[i+1] <= 0;
                    traversal_done_pipe[i+1] <= 0;
                end
            end
            
            // Stage MAX_BST_DEPTH+1: Prepare feedback
            valid_pipe[MAX_BST_DEPTH+1] <= valid_pipe[MAX_BST_DEPTH];
            param0_pipe[MAX_BST_DEPTH+1] <= param0_pipe[MAX_BST_DEPTH];
            param1_pipe[MAX_BST_DEPTH+1] <= param1_pipe[MAX_BST_DEPTH];
            param2_pipe[MAX_BST_DEPTH+1] <= param2_pipe[MAX_BST_DEPTH];
            param3_pipe[MAX_BST_DEPTH+1] <= param3_pipe[MAX_BST_DEPTH];
            param4_pipe[MAX_BST_DEPTH+1] <= param4_pipe[MAX_BST_DEPTH];
            param5_pipe[MAX_BST_DEPTH+1] <= param5_pipe[MAX_BST_DEPTH];
            param6_pipe[MAX_BST_DEPTH+1] <= param6_pipe[MAX_BST_DEPTH];
            param7_pipe[MAX_BST_DEPTH+1] <= param7_pipe[MAX_BST_DEPTH];
            param8_pipe[MAX_BST_DEPTH+1] <= param8_pipe[MAX_BST_DEPTH];
            param9_pipe[MAX_BST_DEPTH+1] <= param9_pipe[MAX_BST_DEPTH];
            
            if (valid_pipe[MAX_BST_DEPTH]) begin
                feedback_base_pipe[0] <= hp_idx_pipe[MAX_BST_DEPTH] * (`PDAQP_N_PARAMETER + 1) * `PDAQP_N_SOLUTION;
            end
            
            // Stage MAX_BST_DEPTH+2: Multiply-accumulate
            valid_pipe[MAX_BST_DEPTH+2] <= valid_pipe[MAX_BST_DEPTH+1];
            
            if (valid_pipe[MAX_BST_DEPTH+1]) begin
                feedback_base_pipe[1] <= feedback_base_pipe[0];
                sol_temp_0[0] <= ($signed(param0_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 0])) + ($signed(param1_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 1])) + ($signed(param2_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 2])) + ($signed(param3_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 3])) + ($signed(param4_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 4])) + ($signed(param5_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 5])) + ($signed(param6_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 6])) + ($signed(param7_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 7])) + ($signed(param8_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 8])) + ($signed(param9_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 0*(`PDAQP_N_PARAMETER+1) + 9]));
                sol_temp_1[0] <= ($signed(param0_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 0])) + ($signed(param1_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 1])) + ($signed(param2_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 2])) + ($signed(param3_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 3])) + ($signed(param4_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 4])) + ($signed(param5_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 5])) + ($signed(param6_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 6])) + ($signed(param7_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 7])) + ($signed(param8_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 8])) + ($signed(param9_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 1*(`PDAQP_N_PARAMETER+1) + 9]));
                sol_temp_2[0] <= ($signed(param0_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 0])) + ($signed(param1_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 1])) + ($signed(param2_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 2])) + ($signed(param3_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 3])) + ($signed(param4_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 4])) + ($signed(param5_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 5])) + ($signed(param6_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 6])) + ($signed(param7_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 7])) + ($signed(param8_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 8])) + ($signed(param9_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 2*(`PDAQP_N_PARAMETER+1) + 9]));
                sol_temp_3[0] <= ($signed(param0_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 0])) + ($signed(param1_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 1])) + ($signed(param2_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 2])) + ($signed(param3_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 3])) + ($signed(param4_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 4])) + ($signed(param5_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 5])) + ($signed(param6_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 6])) + ($signed(param7_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 7])) + ($signed(param8_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 8])) + ($signed(param9_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 3*(`PDAQP_N_PARAMETER+1) + 9]));
                sol_temp_4[0] <= ($signed(param0_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 0])) + ($signed(param1_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 1])) + ($signed(param2_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 2])) + ($signed(param3_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 3])) + ($signed(param4_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 4])) + ($signed(param5_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 5])) + ($signed(param6_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 6])) + ($signed(param7_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 7])) + ($signed(param8_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 8])) + ($signed(param9_pipe[MAX_BST_DEPTH+1]) * $signed(feedbacks[feedback_base_pipe[0] + 4*(`PDAQP_N_PARAMETER+1) + 9]));
            end
            
            // Stage MAX_BST_DEPTH+3: Add offset
            valid_pipe[MAX_BST_DEPTH+3] <= valid_pipe[MAX_BST_DEPTH+2];
            
            if (valid_pipe[MAX_BST_DEPTH+2]) begin
                sol_temp_0[1] <= sol_temp_0[0] + ($signed(feedbacks[feedback_base_pipe[1] + 0*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                sol_temp_1[1] <= sol_temp_1[0] + ($signed(feedbacks[feedback_base_pipe[1] + 1*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                sol_temp_2[1] <= sol_temp_2[0] + ($signed(feedbacks[feedback_base_pipe[1] + 2*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                sol_temp_3[1] <= sol_temp_3[0] + ($signed(feedbacks[feedback_base_pipe[1] + 3*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
                sol_temp_4[1] <= sol_temp_4[0] + ($signed(feedbacks[feedback_base_pipe[1] + 4*(`PDAQP_N_PARAMETER+1) + `PDAQP_N_PARAMETER]) << `FEEDBACK_FRAC_BITS);
            end
            
            // Stage MAX_BST_DEPTH+4: Output
            valid_out <= valid_pipe[MAX_BST_DEPTH+3];
            
            if (valid_pipe[MAX_BST_DEPTH+3]) begin
                sol_out_0 <= sol_temp_0[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
                sol_out_1 <= sol_temp_1[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
                sol_out_2 <= sol_temp_2[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
                sol_out_3 <= sol_temp_3[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
                sol_out_4 <= sol_temp_4[1][`FEEDBACK_FRAC_BITS + `OUTPUT_DATA_WIDTH - 1:`FEEDBACK_FRAC_BITS];
            end
        end
    end
endmodule
