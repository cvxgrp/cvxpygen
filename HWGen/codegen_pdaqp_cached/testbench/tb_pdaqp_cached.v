`timescale 1ns/1ps

module tb_pdqap_solver_cached;

    parameter CLK_PERIOD = 10;
    parameter NUM_TESTS = 10;
    
    // Signals
    reg clk, rst_n;
    reg query_valid;
    reg [31:0] query_params;
    wire query_ready;
    
    wire result_valid;
    wire [1:0] result_region_idx;
    wire [47:0] result_coeffs;
    reg result_ready;
    
    wire [2:0] hp_rom_addr;
    reg [31:0] hp_rom_data;
    wire hp_rom_rd;
    
    wire [1:0] coeff_rom_addr;
    reg [15:0] coeff_rom_data;
    wire coeff_rom_rd;
    
    wire [1:0] neighbor_rom_addr;
    reg [31:0] neighbor_rom_data;
    wire neighbor_rom_rd;
    
    // ROM memories
    reg [31:0] hp_memory [0:4];
    reg [15:0] coeff_memory [0:2];
    
    // DUT
    pdqap_solver_cached #(
        .N_PARAMETERS(2),
        .N_HALFPLANES(5),
        .N_COEFFICIENTS(3),
        .DATA_WIDTH(16),
        .N_REGIONS(3),
        .TREE_DEPTH(2)
    ) dut (.*);
    
    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ROM read logic
    always @(posedge clk) begin
        if (hp_rom_rd) hp_rom_data <= hp_memory[hp_rom_addr];
        if (coeff_rom_rd) coeff_rom_data <= coeff_memory[coeff_rom_addr];
    end
    
    // Test vectors
    reg [31:0] test_vectors [0:NUM_TESTS-1];
    
    initial begin
        test_vectors[0] = 32'h866823b1;
        test_vectors[1] = 32'hb925c669;
        test_vectors[2] = 32'h2d3c3c89;
        test_vectors[3] = 32'h96426465;
        test_vectors[4] = 32'h87a1ec04;
        test_vectors[5] = 32'h015eb7f9;
        test_vectors[6] = 32'hb2e886cc;
        test_vectors[7] = 32'h0b81265e;
        test_vectors[8] = 32'h16dab86f;
        test_vectors[9] = 32'h81aa4f36;
    end
    
    // Test stimulus
    integer i, cycles;
    
    initial begin
        rst_n = 0;
        query_valid = 0;
        result_ready = 1;
        
        // Initialize ROMs
        for (i = 0; i < 5; i = i + 1)
            hp_memory[i] = 0;
        for (i = 0; i < 3; i = i + 1)
            coeff_memory[i] = i[15:0];
        
        $dumpfile("tb_solver.vcd");
        $dumpvars(0, tb_pdqap_solver_cached);
        
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        $display("Starting tests...");
        
        for (i = 0; i < NUM_TESTS; i = i + 1) begin
            @(posedge clk);
            query_params = test_vectors[i];
            query_valid = 1;
            @(posedge clk);
            query_valid = 0;
            
            cycles = 0;
            while (!result_valid && cycles < 1000) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            
            $display("Test %0d: Region=%d, Cycles=%d", i, result_region_idx, cycles);
            @(posedge clk);
        end
        
        repeat(20) @(posedge clk);
        $display("All tests completed!");
        $finish;
    end
    
    initial begin
        #1000000;
        $display("Timeout!");
        $finish;
    end

endmodule
