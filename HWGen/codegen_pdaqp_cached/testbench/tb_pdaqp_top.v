// Testbench for Two-Level Cache
`timescale 1ns/1ps

module tb_pdaqp_top;
    
    localparam N_PARAMETERS = 2;
    localparam N_SOLUTIONS = 3;
    localparam DATA_WIDTH = 32;
    localparam CLK_PERIOD = 10;
    
    reg clk, rst_n, start;
    reg [N_PARAMETERS*DATA_WIDTH-1:0] query;
    wire [N_SOLUTIONS*DATA_WIDTH-1:0] result;
    wire done;
    
    pdaqp_top #(
        .N_PARAMETERS(N_PARAMETERS),
        .N_SOLUTIONS(N_SOLUTIONS),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .query(query),
        .result(result), .done(done)
    );
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    integer test_num, param_idx;
    reg [DATA_WIDTH-1:0] param_value;
    
    initial begin
        $dumpfile("sim/tb.vcd");
        $dumpvars(0, tb_pdaqp_top);
        
        rst_n = 0;
        start = 0;
        query = {(N_PARAMETERS*DATA_WIDTH){1'b0}};
        
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        $display("========================================");
        $display("Two-Level Cache Test Suite");
        $display("N_PARAMETERS=%0d, N_SOLUTIONS=%0d", N_PARAMETERS, N_SOLUTIONS);
        $display("========================================");
        
        for (test_num = 0; test_num < 10; test_num = test_num + 1) begin
            for (param_idx = 0; param_idx < N_PARAMETERS; param_idx = param_idx + 1) begin
                param_value = test_num * 100 + param_idx * 10;
                query[param_idx*DATA_WIDTH +: DATA_WIDTH] = param_value;
            end
            
            $display("\nTest %0d: Query=0x%h", test_num, query);
            
            start = 1;
            #CLK_PERIOD;
            start = 0;
            
            fork
                begin
                    wait(done);
                    $display("  Result=0x%h [PASS]", result);
                end
                begin
                    #(CLK_PERIOD * 200);
                    if (!done) $display("  [TIMEOUT]");
                end
            join_any
            disable fork;
            
            #(CLK_PERIOD*5);
        end
        
        $display("\n========================================");
        $display("All tests completed");
        $display("========================================");
        #(CLK_PERIOD*10);
        $finish;
    end
    
    initial begin
        #(CLK_PERIOD * 5000);
        $display("\nERROR: Global timeout");
        $finish;
    end
endmodule
