`timescale 1ns/1ps
`include "include/pdaqp_config.vh"

module pdaqp_top (
    input wire clk,
    input wire rst_n,
    
    // AXI4-Stream inputs
    input wire [31:0] s_axis_tdata_0,    // {param_0, param_1}
    input wire s_axis_tvalid_0,
    output wire s_axis_tready_0,

    input wire [31:0] s_axis_tdata_1,    // {param_2, param_3}
    input wire s_axis_tvalid_1,
    output wire s_axis_tready_1,

    input wire [31:0] s_axis_tdata_2,    // {param_4, param_5}
    input wire s_axis_tvalid_2,
    output wire s_axis_tready_2,

    input wire [31:0] s_axis_tdata_3,    // {param_6, param_7}
    input wire s_axis_tvalid_3,
    output wire s_axis_tready_3,

    input wire [31:0] s_axis_tdata_4,    // {param_8, param_9}
    input wire s_axis_tvalid_4,
    output wire s_axis_tready_4,
    
    // AXI4-Stream output
    output wire [79:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input wire m_axis_tready
);

    // Extract parameters from AXI
    wire [15:0] param_0 = s_axis_tdata_0[15:0];
    wire [15:0] param_1 = s_axis_tdata_0[31:16];
    wire [15:0] param_2 = s_axis_tdata_1[15:0];
    wire [15:0] param_3 = s_axis_tdata_1[31:16];
    wire [15:0] param_4 = s_axis_tdata_2[15:0];
    wire [15:0] param_5 = s_axis_tdata_2[31:16];
    wire [15:0] param_6 = s_axis_tdata_3[15:0];
    wire [15:0] param_7 = s_axis_tdata_3[31:16];
    wire [15:0] param_8 = s_axis_tdata_4[15:0];
    wire [15:0] param_9 = s_axis_tdata_4[31:16];
    
    wire all_valid = s_axis_tvalid_0 && s_axis_tvalid_1 && s_axis_tvalid_2 && s_axis_tvalid_3 && s_axis_tvalid_4;
    
    wire [15:0] sol_0;
    wire [15:0] sol_1;
    wire [15:0] sol_2;
    wire [15:0] sol_3;
    wire [15:0] sol_4;
    wire bst_valid_out;
    
    pdaqp_bst_lut bst_inst (
        .clk(clk),
        .rst_n(rst_n),
        .param_in_0(param_0),
        .param_in_1(param_1),
        .param_in_2(param_2),
        .param_in_3(param_3),
        .param_in_4(param_4),
        .param_in_5(param_5),
        .param_in_6(param_6),
        .param_in_7(param_7),
        .param_in_8(param_8),
        .param_in_9(param_9),
        .valid_in(all_valid),
        .sol_out_0(sol_0),
        .sol_out_1(sol_1),
        .sol_out_2(sol_2),
        .sol_out_3(sol_3),
        .sol_out_4(sol_4),
        .valid_out(bst_valid_out)
    );
    
    assign m_axis_tdata = {sol_4, sol_3, sol_2, sol_1, sol_0};
    assign m_axis_tvalid = bst_valid_out;
    
    wire ready_all = m_axis_tready;
    
    assign s_axis_tready_0 = ready_all;
    assign s_axis_tready_1 = ready_all;
    assign s_axis_tready_2 = ready_all;
    assign s_axis_tready_3 = ready_all;
    assign s_axis_tready_4 = ready_all;

endmodule
