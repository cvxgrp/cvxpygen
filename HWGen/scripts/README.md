# HWGen - Hardware Generator for Embedded MPC Controllers

Convert C-based MPC controllers to FPGA hardware accelerators automatically.

## Features

- One-click generation from C/H files to Verilog RTL
- Multiple data formats: FP32/FP16 floating-point, FIX16 fixed-point with auto Q-format
- AXI4-Stream interface for system integration
- Pipelined BST solver with configurable IOB limits and latency options

## Quick Start

```bash
git clone https://github.com/your-repo/HWGen.git
cd HWGen
pip install -r requirements.txt

cd scripts
./hwgen quick ../examples/pid_controller.c ../examples/pid_controller.h
```

## Usage

### Quick Generation (Recommended)

```bash
./hwgen quick <c_file> <h_file> [output_dir]
```

### Full Pipeline with Options

```bash
python3 generate.py all -c <c_file> -H <h_file> [options]
```

Options:
- `-o DIR` - Output directory (default: `codegen_<project>`)
- `--fp32` - Use 32-bit floating-point (default: auto FIX16)
- `--fp16` - Use 16-bit floating-point
- `--fixed` - Use manual Q-format fixed-point
- `--max-iob N` - Limit IOB pins (enables data packing)
- `--no-input-buffer` - Disable input buffer (lower latency)
- `-v` - Verbose output

Examples:
```bash
python3 generate.py all -c code/qp.c -H code/qp.h --fp32 --max-iob 200
python3 generate.py all -c code/pid.c -H code/pid.h --fp16 --no-input-buffer -v
```

### Step-by-Step Execution

```bash
# Stage 1: Config generation
python3 generate.py config -c code.c -H code.h -o ./output/include

# Stage 2: Interface generation
python3 generate.py interface -c output/include/code_config.vh -o ./output

# Stage 3: BST solver generation
python3 generate.py bst_fixed output/include/code_config.vh -o ./output/rtl  # Fixed-point
python3 generate.py bst_float output/include/code_config.vh -o ./output/rtl  # Floating-point
```

## Output Structure

```
codegen_<project>/
├── include/
│   ├── <project>_config.vh       # Verilog parameters
│   ├── <project>_timing.vh       # Timing configuration
│   └── *.mem                     # LUT data files
└── rtl/
    ├── <project>_top.v           # AXI4-Stream top module
    └── <project>_bst_lut.v       # BST solver core
```

## Data Format Comparison

| Format | Width | Resource | Use Case |
|--------|-------|----------|----------|
| Auto FIX16 | 16-bit | Low (LUTs only) | Default, recommended |
| Manual FIX16 | 16-bit | Low (LUTs only) | Custom Q-format |
| FP16 | 16-bit | Medium (DSPs) | Balanced precision |
| FP32 | 32-bit | High (DSPs) | Wide dynamic range |

Note: FP32/FP16 require external floating-point operator modules (fp_mult.v, fp_add.v, etc.)

## AXI4-Stream Interface

```verilog
module <project>_top (
    input  wire        aclk,
    input  wire        aresetn,
    
    // Input: state variables
    input  wire [DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    input  wire                  s_axis_tlast,
    
    // Output: control signals
    output wire [DATA_WIDTH-1:0] m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast
);
```

## Utilities

```bash
# List all generation stages
python3 generate.py list

# Clean generated files
./hwgen clean              # Remove all
./hwgen clean <project>    # Remove specific project
```

## Requirements

- Python 3.8+ with numpy, pycparser
- Verilog simulator (Icarus Verilog, ModelSim, etc.)
- FPGA synthesis tools (Vivado, Quartus, etc.)

## Example Workflow

```bash
# Generate hardware
cd scripts
./hwgen quick ../examples/pid_controller.c ../examples/pid_controller.h

# Simulate
cd ../codegen_pid_controller
iverilog -o sim -I include rtl/*.v testbench/tb.v
vvp sim

# Synthesize
vivado -mode batch -source synth.tcl
```

## Troubleshooting

- Use `-v` flag for verbose debug output
- Check `*_config.vh` for generated parameters
- For FP32/FP16: ensure floating-point operator library is included
- For AXI handshake issues: verify both `tvalid` and `tready` signals

## License

MIT License

## Support

- Issues: https://github.com/your-repo/HWGen/issues
- Documentation: https://github.com/your-repo/HWGen/wiki