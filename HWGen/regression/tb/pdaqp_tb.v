`timescale 1ns/1ps
`include "include/pdaqp_config.vh"

module pdaqp_tb;
    localparam CLK_PERIOD = 10;
    
    reg clk;
    reg rst_n;
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD*10);
        rst_n = 1;
        
        #(CLK_PERIOD*100);
        $display("Testbench stub - implement full test");
        $finish;
    end
    
    initial begin
        $dumpfile("pdaqp_tb.vcd");
        $dumpvars(0, pdaqp_tb);
    end
    
endmodule
