// Auto-generated Verilator C++ Testbench for pdaqp
// With Integrated Fixed-Point Golden Model

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vpdaqp_top.h"
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
const int N_PARAMETERS = 10;
const int N_SOLUTIONS = 5;
const int N_HALFPLANES = 352;
const int N_FEEDBACKS = 880;
const int N_TREE_NODES = 131;
const int HALFPLANE_STRIDE = 11;
const int N_AXI_INPUTS = 5;

// Fixed-point format
const int INPUT_FRAC_BITS = 14;
const int OUTPUT_FRAC_BITS = 14;
const int HALFPLANE_FRAC_BITS = 14;
const int FEEDBACK_FRAC_BITS = 14;

// Simulation parameters
const int PIPELINE_DELAY = 14;
const int MAX_CYCLES = 100000;
const int WARMUP_CYCLES = 10;
const int NUM_TEST_VECTORS = 50;

// Fixed-Point Golden Model
class GoldenModel {
private:
    std::vector<int16_t> halfplanes;
    std::vector<int16_t> feedbacks;
    std::vector<uint8_t> hp_list;
    std::vector<uint8_t> jump_list;
    
    template<typename T>
    bool load_mem_file(const std::string& filename, std::vector<T>& data, int expected_size) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open " << filename << std::endl;
            return false;
        }
        
        data.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '/' || line[0] == '#') continue;
            
            unsigned long val = std::stoul(line, nullptr, 16);
            data.push_back(static_cast<T>(val));
        }
        
        if (expected_size > 0 && static_cast<int>(data.size()) != expected_size) {
            std::cerr << "WARNING: " << filename << " size " << data.size() 
                     << " != expected " << expected_size << std::endl;
        }
        
        return true;
    }
    
public:
    GoldenModel() {}
    
    bool load_from_files(const std::string& base_path) {
        std::cout << "Loading golden model data from: " << base_path << "*.mem" << std::endl;
        
        bool ok = true;
        ok &= load_mem_file(base_path + "halfplanes.mem", halfplanes, N_HALFPLANES);
        ok &= load_mem_file(base_path + "feedbacks.mem", feedbacks, N_FEEDBACKS);
        ok &= load_mem_file(base_path + "hp_list.mem", hp_list, N_TREE_NODES);
        ok &= load_mem_file(base_path + "jump_list.mem", jump_list, N_TREE_NODES);
        
        if (ok) {
            std::cout << "  Halfplanes: " << halfplanes.size() << std::endl;
            std::cout << "  Feedbacks:  " << feedbacks.size() << std::endl;
            std::cout << "  HP List:    " << hp_list.size() << std::endl;
            std::cout << "  Jump List:  " << jump_list.size() << std::endl;
        }
        
        return ok;
    }
    
    void evaluate(const int16_t* parameters, int16_t* solutions) {
        // BST traversal
        int id = 0;
        int next_id = id + jump_list[id];
        
        while (next_id != id) {
            int hp_idx = hp_list[id];
            int disp = hp_idx * HALFPLANE_STRIDE;
            
            // Compute halfplane value: Σ(param[i] * coeff[i])
            int32_t val = 0;
            for (int i = 0; i < N_PARAMETERS; i++) {
                val += static_cast<int32_t>(parameters[i]) * static_cast<int32_t>(halfplanes[disp + i]);
            }
            
            // Threshold in same fixed-point format as val
            int32_t threshold = static_cast<int32_t>(halfplanes[disp + N_PARAMETERS]) << HALFPLANE_FRAC_BITS;
            
            // Navigate tree based on decision
            bool decision = (val <= threshold);
            id = next_id + (decision ? 1 : 0);
            next_id = id + jump_list[id];
        }
        
        // Leaf node - evaluate affine function
        int hp_idx = hp_list[id];
        int fb_base = hp_idx * (N_PARAMETERS + 1) * N_SOLUTIONS;
        
        for (int i = 0; i < N_SOLUTIONS; i++) {
            int disp = fb_base + i * (N_PARAMETERS + 1);
            
            int32_t val = 0;
            for (int j = 0; j < N_PARAMETERS; j++) {
                val += static_cast<int32_t>(parameters[j]) * static_cast<int32_t>(feedbacks[disp + j]);
            }
            
            // Add bias term
            val += static_cast<int32_t>(feedbacks[disp + N_PARAMETERS]) << FEEDBACK_FRAC_BITS;
            
            // Extract output bits
            solutions[i] = static_cast<int16_t>((val >> FEEDBACK_FRAC_BITS) & 0xFFFF);
        }
    }
};

// Test Vector
struct TestVector {
    int16_t params[N_PARAMETERS];
    int16_t expected[N_SOLUTIONS];
    std::string description;
};

std::vector<TestVector> generate_test_vectors(GoldenModel& model) {
    std::vector<TestVector> vectors;
    
    auto float_to_fixed = [](float val, int frac_bits) -> int16_t {
        int32_t fixed = static_cast<int32_t>(std::round(val * (1 << frac_bits)));
        if (fixed > 32767) fixed = 32767;
        if (fixed < -32768) fixed = -32768;
        return static_cast<int16_t>(fixed);
    };
    
    float max_val = (32767.0f) / (1 << INPUT_FRAC_BITS);
    float min_val = (-32768.0f) / (1 << INPUT_FRAC_BITS);
    
    std::cout << "\nGenerating " << NUM_TEST_VECTORS << " test vectors..." << std::endl;
    std::cout << "  Input range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    std::srand(42);
    
    // Test 1: All zeros
    {
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) tv.params[i] = 0;
        model.evaluate(tv.params, tv.expected);
        tv.description = "All zeros";
        vectors.push_back(tv);
    }
    
    // Test 2-4: Boundary values
    for (int test = 0; test < 3; test++) {
        TestVector tv;
        float val = (test == 0) ? max_val : (test == 1) ? min_val : max_val * 0.5f;
        for (int i = 0; i < N_PARAMETERS; i++) {
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }
        model.evaluate(tv.params, tv.expected);
        tv.description = (test == 0) ? "All maximum" : (test == 1) ? "All minimum" : "All half max";
        vectors.push_back(tv);
    }
    
    // Test 5-9: Pattern tests
    for (int test = 0; test < 5; test++) {
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) {
            float val = 0.0f;
            if (test == 0) {
                val = (i % 2 == 0) ? max_val : min_val;
            } else if (test == 1) {
                val = min_val + (max_val - min_val) * i / (N_PARAMETERS - 1);
            } else if (test == 2) {
                val = 0.1f * ((i % 2 == 0) ? 1.0f : -1.0f);
            } else if (test == 3) {
                float p = std::pow(2.0f, -5 + i);
                val = std::min(max_val, std::max(min_val, p));
            } else {
                val = max_val * 0.7f * std::sin(i * 3.14159f / 4.0f);
            }
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }
        model.evaluate(tv.params, tv.expected);
        tv.description = "Pattern " + std::to_string(test + 1);
        vectors.push_back(tv);
    }
    
    // Test 10+: Random tests
    for (int test = 0; test < NUM_TEST_VECTORS - 9; test++) {
        TestVector tv;
        for (int i = 0; i < N_PARAMETERS; i++) {
            float range = max_val - min_val;
            float val = min_val + range * (std::rand() / static_cast<float>(RAND_MAX));
            tv.params[i] = float_to_fixed(val, INPUT_FRAC_BITS);
        }
        model.evaluate(tv.params, tv.expected);
        tv.description = "Random " + std::to_string(test + 1);
        vectors.push_back(tv);
    }
    
    std::cout << "✓ Generated " << vectors.size() << " test vectors" << std::endl;
    return vectors;
}

// Testbench Class
class PDACPTestbench {
private:
    Vpdaqp_top* dut;
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
    PDACPTestbench() {
        dut = new Vpdaqp_top;
        
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open("pdaqp_sim.vcd");
        
        sim_time = 0;
        cycle_count = 0;
        tests_passed = 0;
        tests_failed = 0;
        output_count = 0;
        
        log_file.open("pdaqp_test_log.txt");
        
        if (!golden_model.load_from_files("include/pdaqp_")) {
            std::cerr << "FATAL: Failed to load golden model data" << std::endl;
            std::cerr << "Expected .mem files at: include/pdaqp_*.mem" << std::endl;
            std::cerr << "Make sure to run the executable from the project root directory" << std::endl;
            exit(1);
        }
        
        test_vectors = generate_test_vectors(golden_model);
    }
    
    ~PDACPTestbench() {
        tfp->close();
        delete tfp;
        delete dut;
        log_file.close();
    }
    
    void log(const std::string& msg) {
        std::string formatted = "[" + std::to_string(cycle_count) + "] " + msg;
        std::cout << formatted << std::endl;
        log_file << formatted << std::endl;
    }
    
    void clock_tick() {
        dut->clk = 0;
        dut->eval();
        tfp->dump(sim_time++);
        
        dut->clk = 1;
        dut->eval();
        tfp->dump(sim_time++);
        
        cycle_count++;
    }
    
    void reset() {
        log("=== Reset Sequence ===");
        dut->rst_n = 0;
        dut->m_axis_tready = 1;
        
        dut->s_axis_tvalid_0 = 0;
        dut->s_axis_tvalid_1 = 0;
        dut->s_axis_tvalid_2 = 0;
        dut->s_axis_tvalid_3 = 0;
        dut->s_axis_tvalid_4 = 0;
        
        for (int i = 0; i < WARMUP_CYCLES; i++) clock_tick();
        
        dut->rst_n = 1;
        log("Reset released");
        
        for (int i = 0; i < WARMUP_CYCLES; i++) clock_tick();
    }
    
    void send_problem(const int16_t params[N_PARAMETERS]) {
        dut->s_axis_tdata_0 = (static_cast<uint32_t>(params[1]) << 16) | (params[0] & 0xFFFF);
        dut->s_axis_tdata_1 = (static_cast<uint32_t>(params[3]) << 16) | (params[2] & 0xFFFF);
        dut->s_axis_tdata_2 = (static_cast<uint32_t>(params[5]) << 16) | (params[4] & 0xFFFF);
        dut->s_axis_tdata_3 = (static_cast<uint32_t>(params[7]) << 16) | (params[6] & 0xFFFF);
        dut->s_axis_tdata_4 = (static_cast<uint32_t>(params[9]) << 16) | (params[8] & 0xFFFF);
        
        dut->s_axis_tvalid_0 = 1;
        dut->s_axis_tvalid_1 = 1;
        dut->s_axis_tvalid_2 = 1;
        dut->s_axis_tvalid_3 = 1;
        dut->s_axis_tvalid_4 = 1;
        
        int timeout = 0;
        while (!(dut->s_axis_tready_0 && dut->s_axis_tready_1 && dut->s_axis_tready_2 && dut->s_axis_tready_3 && dut->s_axis_tready_4) && timeout++ < 100) {
            clock_tick();
        }
        
        clock_tick();
        
        dut->s_axis_tvalid_0 = 0;
        dut->s_axis_tvalid_1 = 0;
        dut->s_axis_tvalid_2 = 0;
        dut->s_axis_tvalid_3 = 0;
        dut->s_axis_tvalid_4 = 0;
    }

    
    void extract_output(uint16_t output[N_SOLUTIONS]) {
        output[0] = (dut->m_axis_tdata[0] >> 0) & 0xFFFF;
        output[1] = (dut->m_axis_tdata[0] >> 16) & 0xFFFF;
        output[2] = (dut->m_axis_tdata[1] >> 0) & 0xFFFF;
        output[3] = (dut->m_axis_tdata[1] >> 16) & 0xFFFF;
        output[4] = (dut->m_axis_tdata[2] >> 0) & 0xFFFF;
    }

    
    bool check_output(const TestVector& tv) {
        if (!dut->m_axis_tvalid) return false;
        
        uint16_t actual[N_SOLUTIONS];
        extract_output(actual);
        
        bool pass = true;
        int max_error = 0;
        
        for (int i = 0; i < N_SOLUTIONS; i++) {
            int16_t diff = std::abs(static_cast<int16_t>(actual[i]) - 
                                   static_cast<int16_t>(tv.expected[i]));
            if (diff > max_error) max_error = diff;
            if (diff > 2) pass = false;
        }
        
        if (pass) {
            tests_passed++;
            log("✓ PASS: " + tv.description + " (max_err=" + std::to_string(max_error) + ")");
        } else {
            tests_failed++;
            log("✗ FAIL: " + tv.description);
            
            log_file << "  Expected: ";
            for (int i = 0; i < N_SOLUTIONS; i++) {
                log_file << "0x" << std::hex << tv.expected[i] << " ";
            }
            log_file << std::dec << std::endl;
            
            log_file << "  Actual:   ";
            for (int i = 0; i < N_SOLUTIONS; i++) {
                log_file << "0x" << std::hex << actual[i] << " ";
            }
            log_file << std::dec << std::endl;
        }
        
        output_count++;
        return true;
    }
    
    void run_tests() {
        log("\n=== Running Test Suite ===");
        log("Total tests: " + std::to_string(test_vectors.size()));
        
        for (size_t i = 0; i < test_vectors.size(); i++) {
            send_problem(test_vectors[i].params);
            
            bool got_output = false;
            for (int j = 0; j < PIPELINE_DELAY + 20; j++) {
                clock_tick();
                if (check_output(test_vectors[i])) {
                    got_output = true;
                    break;
                }
            }
            
            if (!got_output) {
                log("✗ TIMEOUT: " + test_vectors[i].description);
                tests_failed++;
            }
            
            if ((i + 1) % 10 == 0) {
                log("Progress: " + std::to_string(i + 1) + "/" + 
                    std::to_string(test_vectors.size()));
            }
        }
        
        for (int i = 0; i < PIPELINE_DELAY * 2; i++) clock_tick();
    }
    
    void print_results() {
        log("\n" + std::string(70, '='));
        log("TEST RESULTS");
        log(std::string(70, '='));
        log("Total tests:  " + std::to_string(test_vectors.size()));
        log("Passed:       " + std::to_string(tests_passed));
        log("Failed:       " + std::to_string(tests_failed));
        
        float pass_rate = 100.0f * tests_passed / test_vectors.size();
        log("Pass rate:    " + std::to_string(pass_rate) + "%");
        log("Total cycles: " + std::to_string(cycle_count));
        log(std::string(70, '='));
        
        if (tests_failed == 0) {
            log("\n✓✓✓ ALL TESTS PASSED ✓✓✓");
        } else {
            log("\n✗✗✗ SOME TESTS FAILED ✗✗✗");
        }
    }
};

// Main
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  PDAQP Verilator Testbench" << std::endl;
    std::cout << "  With Integrated Fixed-Point Golden Model" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto tb = new PDACPTestbench();
    tb->reset();
    tb->run_tests();
    tb->print_results();
    delete tb;
    
    std::cout << "\nSimulation complete." << std::endl;
    std::cout << "Files: pdaqp_sim.vcd, pdaqp_test_log.txt" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
