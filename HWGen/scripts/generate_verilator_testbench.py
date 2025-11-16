#!/usr/bin/env python3
"""
Generate Verilator C++ Testbench with Integrated Golden Model
"""

import argparse
import re
from pathlib import Path
import sys
import random
import math

class VerilatorTestbenchGenerator:
    """Generate self-contained testbench with golden model"""
    
    def __init__(self, config_file, include_dir, output_dir):
        self.config_file = Path(config_file)
        self.include_dir = Path(include_dir)
        self.output_dir = Path(output_dir)
        
        # Parse configuration
        self.config = self.parse_config()
        self.project_name = self.extract_project_name()
        
        # Calculate derived parameters
        self.n_param = self.config['n_parameters']
        self.n_sol = self.config['n_solutions']
        self.params_per_interface = 2
        self.n_axi_inputs = (self.n_param + self.params_per_interface - 1) // self.params_per_interface
        self.output_width = self.n_sol * self.config['output_width']
        self.output_words = (self.output_width + 31) // 32
        
        # Determine Verilator signal type based on bit width
        if self.output_width <= 32:
            self.output_type = "IData"
        elif self.output_width <= 64:
            self.output_type = "QData"
        else:
            self.output_type = "WData"
        
        self.print_config_summary()
    
    def print_config_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*70}")
        print(f"Verilator Testbench Generator: {self.project_name}")
        print(f"With Integrated Fixed-Point Golden Model")
        print(f"{'='*70}")
        print(f"Problem Configuration:")
        print(f"  Parameters:              {self.n_param}")
        print(f"  Solutions:               {self.n_sol}")
        print(f"  Halfplanes:              {self.config['n_halfplanes']}")
        print(f"  Feedbacks:               {self.config['n_feedbacks']}")
        print(f"  Tree Nodes:              {self.config['n_tree_nodes']}")
        print(f"  BST Depth:               {self.config['bst_depth']}")
        print(f"\nFixed-Point Precision:")
        print(f"  Input:                   Q{self.config['input_int']}.{self.config['input_frac']} (16-bit)")
        print(f"  Output:                  Q{self.config['output_int']}.{self.config['output_frac']} (16-bit)")
        print(f"  Halfplane fractional:    {self.config['halfplane_frac']} bits")
        print(f"  Feedback fractional:     {self.config['feedback_frac']} bits")
        print(f"\nInterface:")
        print(f"  AXI Inputs:              {self.n_axi_inputs}")
        print(f"  Output Width:            {self.output_width} bits")
        print(f"  Output Type:             {self.output_type}")
        print(f"{'='*70}\n")
    
    def parse_config(self):
        """Parse Verilog config file"""
        config = {}
        
        with open(self.config_file, 'r') as f:
            content = f.read()
        
        patterns = {
            'n_parameters': r'`define\s+PDAQP_N_PARAMETER\s+(\d+)',
            'n_solutions': r'`define\s+PDAQP_N_SOLUTION\s+(\d+)',
            'n_halfplanes': r'`define\s+PDAQP_HALFPLANES\s+(\d+)',
            'n_feedbacks': r'`define\s+PDAQP_FEEDBACKS\s+(\d+)',
            'n_tree_nodes': r'`define\s+PDAQP_TREE_NODES\s+(\d+)',
            'bst_depth': r'`define\s+PDAQP_ESTIMATED_BST_DEPTH\s+(\d+)',
            'halfplane_stride': r'`define\s+PDAQP_HALFPLANE_STRIDE\s+(\d+)',
            'input_width': r'`define\s+INPUT_DATA_WIDTH\s+(\d+)',
            'output_width': r'`define\s+OUTPUT_DATA_WIDTH\s+(\d+)',
            'input_int': r'`define\s+INPUT_INT_BITS\s+(\d+)',
            'input_frac': r'`define\s+INPUT_FRAC_BITS\s+(\d+)',
            'output_int': r'`define\s+OUTPUT_INT_BITS\s+(\d+)',
            'output_frac': r'`define\s+OUTPUT_FRAC_BITS\s+(\d+)',
            'halfplane_frac': r'`define\s+HALFPLANE_FRAC_BITS\s+(\d+)',
            'feedback_frac': r'`define\s+FEEDBACK_FRAC_BITS\s+(\d+)',
        }
        
        # Parse all patterns
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            config[key] = int(match.group(1)) if match else 0
        
        # Set defaults for missing values
        if config['input_width'] == 0:
            config['input_width'] = 16
        if config['output_width'] == 0:
            config['output_width'] = 16
        if config['input_int'] == 0:
            config['input_int'] = 2
        if config['input_frac'] == 0:
            config['input_frac'] = 14
        if config['output_int'] == 0:
            config['output_int'] = 2
        if config['output_frac'] == 0:
            config['output_frac'] = 14
        if config['halfplane_frac'] == 0:
            config['halfplane_frac'] = 14
        if config['feedback_frac'] == 0:
            config['feedback_frac'] = 13
        if config['bst_depth'] == 0:
            config['bst_depth'] = 7
        
        # Calculate halfplane_stride if not defined
        if config['halfplane_stride'] == 0:
            config['halfplane_stride'] = config['n_parameters'] + 1
        
        # Verify required fields
        if config['n_parameters'] == 0:
            raise ValueError("PDAQP_N_PARAMETER not found in config file")
        if config['n_solutions'] == 0:
            raise ValueError("PDAQP_N_SOLUTION not found in config file")
        
        return config

    def extract_project_name(self):
        """Extract project name from config file"""
        filename = self.config_file.stem
        if filename.endswith('_config'):
            return filename[:-7]
        return self.config_file.parent.parent.name or 'pdaqp'
    
    def find_mem_files(self):
        """Find and verify memory initialization files"""
        mem_files = {
            'halfplanes': self.include_dir / f"{self.project_name}_halfplanes.mem",
            'feedbacks': self.include_dir / f"{self.project_name}_feedbacks.mem",
            'hp_list': self.include_dir / f"{self.project_name}_hp_list.mem",
            'jump_list': self.include_dir / f"{self.project_name}_jump_list.mem",
        }
        
        print("Memory Files:")
        for name, path in mem_files.items():
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {name:12s}: {path}")
        
        # Check all exist
        missing = [name for name, path in mem_files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing memory files: {', '.join(missing)}")
        
        return mem_files
    
    def generate_testbench(self):
        """Generate complete C++ testbench with golden model"""
        
        mem_files = self.find_mem_files()
        pipeline_delay = self.config['bst_depth'] + 4
        
        # Use relative path from project root to include directory
        mem_path_prefix = f"include/{self.project_name}_"
        
        cpp_code = f'''// Auto-generated Verilator C++ Testbench for {self.project_name}
// With Integrated Fixed-Point Golden Model

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "V{self.project_name}_top.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <cstdlib>
#include <ctime>

// Configuration (extracted from config.vh)
const int N_PARAMETERS = {self.n_param};
const int N_SOLUTIONS = {self.n_sol};
const int N_HALFPLANES = {self.config['n_halfplanes']};
const int N_FEEDBACKS = {self.config['n_feedbacks']};
const int N_TREE_NODES = {self.config['n_tree_nodes']};
const int HALFPLANE_STRIDE = {self.config['halfplane_stride']};
const int N_AXI_INPUTS = {self.n_axi_inputs};

// Fixed-point format
const int INPUT_FRAC_BITS = {self.config['input_frac']};
const int OUTPUT_FRAC_BITS = {self.config['output_frac']};
const int HALFPLANE_FRAC_BITS = {self.config['halfplane_frac']};
const int FEEDBACK_FRAC_BITS = {self.config['feedback_frac']};

// Simulation parameters
const int PIPELINE_DELAY = {pipeline_delay};
const int MAX_CYCLES = 100000;
const int WARMUP_CYCLES = 10;
const int NUM_TEST_VECTORS = 50;

// Fixed-Point Golden Model
class GoldenModel {{
private:
    std::vector<int16_t> halfplanes;
    std::vector<int16_t> feedbacks;
    std::vector<uint8_t> hp_list;
    std::vector<uint8_t> jump_list;
    
    template<typename T>
    bool load_mem_file(const std::string& filename, std::vector<T>& data, int expected_size) {{
        std::ifstream file(filename);
        if (!file.is_open()) {{
            std::cerr << "ERROR: Cannot open " << filename << std::endl;
            return false;
        }}
        
        data.clear();
        std::string line;
        while (std::getline(file, line)) {{
            if (line.empty() || line[0] == '/' || line[0] == '#') continue;
            
            unsigned long val = std::stoul(line, nullptr, 16);
            data.push_back(static_cast<T>(val));
        }}
        
        if (expected_size > 0 && static_cast<int>(data.size()) != expected_size) {{
            std::cerr << "WARNING: " << filename << " size " << data.size() 
                     << " != expected " << expected_size << std::endl;
        }}
        
        return true;
    }}
    
public:
    GoldenModel() {{}}
    
    bool load_from_files(const std::string& base_path) {{
        std::cout << "Loading golden model data from: " << base_path << "*.mem" << std::endl;
        
        bool ok = true;
        ok &= load_mem_file(base_path + "halfplanes.mem", halfplanes, N_HALFPLANES);
        ok &= load_mem_file(base_path + "feedbacks.mem", feedbacks, N_FEEDBACKS);
        ok &= load_mem_file(base_path + "hp_list.mem", hp_list, N_TREE_NODES);
        ok &= load_mem_file(base_path + "jump_list.mem", jump_list, N_TREE_NODES);
        
        if (ok) {{
            std::cout << "  Halfplanes: " << halfplanes.size() << std::endl;
            std::cout << "  Feedbacks:  " << feedbacks.size() << std::endl;
            std::cout << "  HP List:    " << hp_list.size() << std::endl;
            std::cout << "  Jump List:  " << jump_list.size() << std::endl;
        }}
        
        return ok;
    }}
    
    void evaluate(const int16_t* parameters, int16_t* solutions) {{
        // BST traversal
        int id = 0;
        int next_id = id + jump_list[id];
        
        while (next_id != id) {{
            int hp_idx = hp_list[id];
            int disp = hp_idx * HALFPLANE_STRIDE;
            
            // Compute halfplane value: Σ(param[i] * coeff[i])
            int32_t val = 0;
            for (int i = 0; i < N_PARAMETERS; i++) {{
                val += static_cast<int32_t>(parameters[i]) * static_cast<int32_t>(halfplanes[disp + i]);
            }}
            
            // Threshold in same fixed-point format as val
            int32_t threshold = static_cast<int32_t>(halfplanes[disp + N_PARAMETERS]) << HALFPLANE_FRAC_BITS;
            
            // Navigate tree based on decision
            bool decision = (val <= threshold);
            id = next_id + (decision ? 1 : 0);
            next_id = id + jump_list[id];
        }}
        
        // Leaf node - evaluate affine function
        int hp_idx = hp_list[id];
        int fb_base = hp_idx * (N_PARAMETERS + 1) * N_SOLUTIONS;
        
        for (int i = 0; i < N_SOLUTIONS; i++) {{
            int disp = fb_base + i * (N_PARAMETERS + 1);
            
            int32_t val = 0;
            for (int j = 0; j < N_PARAMETERS; j++) {{
                val += static_cast<int32_t>(parameters[j]) * static_cast<int32_t>(feedbacks[disp + j]);
            }}
            
            // Add bias term
            val += static_cast<int32_t>(feedbacks[disp + N_PARAMETERS]) << FEEDBACK_FRAC_BITS;
            
            // Extract output bits
            solutions[i] = static_cast<int16_t>((val >> FEEDBACK_FRAC_BITS) & 0xFFFF);
        }}
    }}
}};

// Test Vector
struct TestVector {{
    int16_t params[N_PARAMETERS];
    int16_t expected[N_SOLUTIONS];
    std::string description;
}};

std::vector<TestVector> generate_test_vectors(GoldenModel& model) {{
    std::vector<TestVector> vectors;
    
    auto float_to_fixed = [](float val, int frac_bits) -> int16_t {{
        int32_t fixed = static_cast<int32_t>(std::round(val * (1 << frac_bits)));
        if (fixed > 32767) fixed = 32767;
        if (fixed < -32768) fixed = -32768;
        return static_cast<int16_t>(fixed);
    }};
    
    float max_val = (32767.0f) / (1 << INPUT_FRAC_BITS);
    float min_val = (-32768.0f) / (1 << INPUT_FRAC_BITS);
    
    std::cout << "\\nGenerating " << NUM_TEST_VECTORS << " test vectors..." << std::endl;
    std::cout << "  Input range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    std::srand(42);
    
    // Test 1: All zeros
    {{
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) tv.params[i] = 0;
        model.evaluate(tv.params, tv.expected);
        tv.description = "All zeros";
        vectors.push_back(tv);
    }}
    
    // Test 2-4: Boundary values
    for (int test = 0; test < 3; test++) {{
        TestVector tv;
        float val = (test == 0) ? max_val : (test == 1) ? min_val : max_val * 0.5f;
        for (int i = 0; i < N_PARAMETERS; i++) {{
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }}
        model.evaluate(tv.params, tv.expected);
        tv.description = (test == 0) ? "All maximum" : (test == 1) ? "All minimum" : "All half max";
        vectors.push_back(tv);
    }}
    
    // Test 5-9: Pattern tests
    for (int test = 0; test < 5; test++) {{
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) {{
            float val = 0.0f;
            if (test == 0) {{
                val = (i % 2 == 0) ? max_val : min_val;
            }} else if (test == 1) {{
                val = min_val + (max_val - min_val) * i / (N_PARAMETERS - 1);
            }} else if (test == 2) {{
                val = 0.1f * ((i % 2 == 0) ? 1.0f : -1.0f);
            }} else if (test == 3) {{
                float p = std::pow(2.0f, -5 + i);
                val = std::min(max_val, std::max(min_val, p));
            }} else {{
                val = max_val * 0.7f * std::sin(i * 3.14159f / 4.0f);
            }}
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }}
        model.evaluate(tv.params, tv.expected);
        tv.description = "Pattern " + std::to_string(test + 1);
        vectors.push_back(tv);
    }}
    
    // Test 10+: Random tests
    for (int test = 0; test < NUM_TEST_VECTORS - 9; test++) {{
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) {{
            float range = max_val - min_val;
            float val = min_val + range * (std::rand() / static_cast<float>(RAND_MAX));
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }}
        model.evaluate(tv.params, tv.expected);
        tv.description = "Random " + std::to_string(test + 1);
        vectors.push_back(tv);
    }}
    
    std::cout << "✓ Generated " << vectors.size() << " test vectors" << std::endl;
    return vectors;
}}

// Testbench Class
class PDACPTestbench {{
private:
    V{self.project_name}_top* dut;
    VerilatedVcdC* tfp;
    uint64_t sim_time;
    uint64_t cycle_count;
    std::ofstream log_file;
    
    GoldenModel golden_model;
    std::vector<TestVector> test_vectors;
    
    int tests_passed;
    int tests_failed;
    int output_count;
    
public:
    PDACPTestbench() {{
        dut = new V{self.project_name}_top;
        
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open("{self.project_name}_sim.vcd");
        
        sim_time = 0;
        cycle_count = 0;
        tests_passed = 0;
        tests_failed = 0;
        output_count = 0;
        
        log_file.open("{self.project_name}_test_log.txt");
        
        if (!golden_model.load_from_files("{mem_path_prefix}")) {{
            std::cerr << "FATAL: Failed to load golden model data" << std::endl;
            std::cerr << "Expected .mem files at: {mem_path_prefix}*.mem" << std::endl;
            std::cerr << "Make sure to run the executable from the project root directory" << std::endl;
            exit(1);
        }}
        
        test_vectors = generate_test_vectors(golden_model);
    }}
    
    ~PDACPTestbench() {{
        tfp->close();
        delete tfp;
        delete dut;
        log_file.close();
    }}
    
    void log(const std::string& msg) {{
        std::string formatted = "[" + std::to_string(cycle_count) + "] " + msg;
        std::cout << formatted << std::endl;
        log_file << formatted << std::endl;
    }}
    
    void clock_tick() {{
        dut->clk = 0;
        dut->eval();
        tfp->dump(sim_time++);
        
        dut->clk = 1;
        dut->eval();
        tfp->dump(sim_time++);
        
        cycle_count++;
    }}
    
    void reset() {{
        log("=== Reset Sequence ===");
        dut->rst_n = 0;
        dut->m_axis_tready = 1;
        
{self.generate_reset_code()}
        
        for (int i = 0; i < WARMUP_CYCLES; i++) clock_tick();
        
        dut->rst_n = 1;
        log("Reset released");
        
        for (int i = 0; i < WARMUP_CYCLES; i++) clock_tick();
    }}
    
{self.generate_send_problem_function()}
    
{self.generate_extract_output_function()}
    
    bool check_output(const TestVector& tv) {{
        if (!dut->m_axis_tvalid) return false;
        
        uint16_t actual[N_SOLUTIONS];
        extract_output(actual);
        
        bool pass = true;
        int max_error = 0;
        
        for (int i = 0; i < N_SOLUTIONS; i++) {{
            int16_t diff = std::abs(static_cast<int16_t>(actual[i]) - 
                                   static_cast<int16_t>(tv.expected[i]));
            if (diff > max_error) max_error = diff;
            if (diff > 2) pass = false;
        }}
        
        if (pass) {{
            tests_passed++;
            log("✓ PASS: " + tv.description + " (max_err=" + std::to_string(max_error) + ")");
        }} else {{
            tests_failed++;
            log("✗ FAIL: " + tv.description);
            
            log_file << "  Expected: ";
            for (int i = 0; i < N_SOLUTIONS; i++) {{
                log_file << "0x" << std::hex << tv.expected[i] << " ";
            }}
            log_file << std::dec << std::endl;
            
            log_file << "  Actual:   ";
            for (int i = 0; i < N_SOLUTIONS; i++) {{
                log_file << "0x" << std::hex << actual[i] << " ";
            }}
            log_file << std::dec << std::endl;
        }}
        
        output_count++;
        return true;
    }}
    
    void run_tests() {{
        log("\\n=== Running Test Suite ===");
        log("Total tests: " + std::to_string(test_vectors.size()));
        
        for (size_t i = 0; i < test_vectors.size(); i++) {{
            send_problem(test_vectors[i].params);
            
            bool got_output = false;
            for (int j = 0; j < PIPELINE_DELAY + 20; j++) {{
                clock_tick();
                if (check_output(test_vectors[i])) {{
                    got_output = true;
                    break;
                }}
            }}
            
            if (!got_output) {{
                log("✗ TIMEOUT: " + test_vectors[i].description);
                tests_failed++;
            }}
            
            if ((i + 1) % 10 == 0) {{
                log("Progress: " + std::to_string(i + 1) + "/" + 
                    std::to_string(test_vectors.size()));
            }}
        }}
        
        for (int i = 0; i < PIPELINE_DELAY * 2; i++) clock_tick();
    }}
    
    void print_results() {{
        log("\\n" + std::string(70, '='));
        log("TEST RESULTS");
        log(std::string(70, '='));
        log("Total tests:  " + std::to_string(test_vectors.size()));
        log("Passed:       " + std::to_string(tests_passed));
        log("Failed:       " + std::to_string(tests_failed));
        
        float pass_rate = 100.0f * tests_passed / test_vectors.size();
        log("Pass rate:    " + std::to_string(pass_rate) + "%");
        log("Total cycles: " + std::to_string(cycle_count));
        log(std::string(70, '='));
        
        if (tests_failed == 0) {{
            log("\\n✓✓✓ ALL TESTS PASSED ✓✓✓");
        }} else {{
            log("\\n✗✗✗ SOME TESTS FAILED ✗✗✗");
        }}
    }}
}};

// Main
int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    
    std::cout << "\\n" << std::string(70, '=') << std::endl;
    std::cout << "  {self.project_name.upper()} Verilator Testbench" << std::endl;
    std::cout << "  With Integrated Fixed-Point Golden Model" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto tb = new PDACPTestbench();
    tb->reset();
    tb->run_tests();
    tb->print_results();
    delete tb;
    
    std::cout << "\\nSimulation complete." << std::endl;
    std::cout << "Files: {self.project_name}_sim.vcd, {self.project_name}_test_log.txt" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}}
'''
        return cpp_code
    
    def generate_reset_code(self):
        """Generate reset initialization for all AXI input channels"""
        lines = []
        for i in range(self.n_axi_inputs):
            lines.append(f"        dut->s_axis_tvalid_{i} = 0;")
        return '\n'.join(lines)
    
    def generate_send_problem_function(self):
        """Generate function to pack parameters and send via AXI interfaces"""
        packing_lines = []
        for i in range(self.n_axi_inputs):
            p0_idx = i * self.params_per_interface
            p1_idx = p0_idx + 1
            if p1_idx < self.n_param:
                packing_lines.append(
                    f"        dut->s_axis_tdata_{i} = "
                    f"(static_cast<uint32_t>(params[{p1_idx}]) << 16) | "
                    f"(params[{p0_idx}] & 0xFFFF);"
                )
            else:
                packing_lines.append(
                    f"        dut->s_axis_tdata_{i} = params[{p0_idx}] & 0xFFFF;"
                )
        
        valid_lines = []
        ready_checks = []
        for i in range(self.n_axi_inputs):
            valid_lines.append(f"        dut->s_axis_tvalid_{i} = 1;")
            ready_checks.append(f"dut->s_axis_tready_{i}")
        
        ready_condition = " && ".join(ready_checks)
        
        deassert_lines = []
        for i in range(self.n_axi_inputs):
            deassert_lines.append(f"        dut->s_axis_tvalid_{i} = 0;")
        
        return f'''    void send_problem(const int16_t params[N_PARAMETERS]) {{
{chr(10).join(packing_lines)}
        
{chr(10).join(valid_lines)}
        
        int timeout = 0;
        while (!({ready_condition}) && timeout++ < 100) {{
            clock_tick();
        }}
        
        clock_tick();
        
{chr(10).join(deassert_lines)}
    }}
'''
    
    def generate_extract_output_function(self):
        """Generate function to extract solutions from output"""
        
        if self.output_width <= 64:
            return self._generate_extract_scalar()
        else:
            return self._generate_extract_wdata()
    
    def _generate_extract_scalar(self):
        """Generate extraction for ≤64 bit signals"""
        extract_lines = []
        for i in range(self.n_sol):
            shift = i * 16
            extract_lines.append(
                f"        output[{i}] = (dut->m_axis_tdata >> {shift}) & 0xFFFF;"
            )
        
        return f'''    void extract_output(uint16_t output[N_SOLUTIONS]) {{
{chr(10).join(extract_lines)}
    }}
'''
    
    def _generate_extract_wdata(self):
        """Generate extraction for >64 bit signals (array access)"""
        extract_lines = []
        for i in range(self.n_sol):
            bit_offset = i * 16
            word_idx = bit_offset // 32
            bit_in_word = bit_offset % 32
            
            if bit_in_word <= 16:
                extract_lines.append(
                    f"        output[{i}] = (dut->m_axis_tdata[{word_idx}] >> {bit_in_word}) & 0xFFFF;"
                )
            else:
                extract_lines.append(
                    f"        output[{i}] = ((dut->m_axis_tdata[{word_idx}] >> {bit_in_word}) | "
                    f"(dut->m_axis_tdata[{word_idx+1}] << {32-bit_in_word})) & 0xFFFF;"
                )
        
        return f'''    void extract_output(uint16_t output[N_SOLUTIONS]) {{
{chr(10).join(extract_lines)}
    }}
'''
    
    def generate_all(self):
        """Main generation entry point"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cpp_code = self.generate_testbench()
        cpp_file = self.output_dir / "sim_main.cpp"
        
        with open(cpp_file, 'w') as f:
            f.write(cpp_code)
        
        print(f"\n✓ Generated: {cpp_file}")
        print(f"\n{'='*70}")
        print("✓ Testbench generation complete!")
        print(f"{'='*70}")
        print(f"\nNext steps:")
        print(f"  1. Compile with Verilator")
        print(f"  2. Run simulation from project root directory")
        print(f"  3. Check output: {self.project_name}_sim.vcd, {self.project_name}_test_log.txt")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Generate Verilator testbench with integrated golden model'
    )
    parser.add_argument('config_file', help='Configuration .vh file')
    parser.add_argument('-i', '--include', required=True,
                       help='Include directory with .mem files')
    parser.add_argument('-o', '--output', default='c_tb',
                       help='Output directory (default: c_tb)')
    
    args = parser.parse_args()
    
    try:
        generator = VerilatorTestbenchGenerator(
            args.config_file,
            args.include,
            args.output
        )
        generator.generate_all()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()