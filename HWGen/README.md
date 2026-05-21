# PDAQP Hardware Generator

**Automated RTL generation and verification for PDAQP controllers**

---

## Quick Start

```bash
# Positional syntax
make complete my_project/pdaqp.c my_project/pdaqp.h
make sim_compile my_project
make sim_run my_project

# Named syntax
make complete C_FILE=my.c H_FILE=my.h
make sim_compile WORK_DIR=my_project
```

---

## Installation

### Verilator (Fast Open-Source Simulator)

#### Ubuntu/Debian

```bash
sudo apt-get install -y verilator gtkwave
verilator --version
make check
```

#### Latest from Source

```bash
# Install dependencies
sudo apt-get install -y git make autoconf g++ flex bison

# Build Verilator
git clone https://github.com/verilator/verilator
cd verilator
git checkout stable
autoconf && ./configure && make -j$(nproc)
sudo make install

# Verify
verilator --version
```

---

## Usage

### Complete Workflows

```bash
# Generate everything + simulate
make complete_with_sim my.c my.h

# Fast regeneration from config
make hardware_with_sim my_project/include/pdaqp_config.vh

# Named syntax
make complete_with_sim C_FILE=my.c H_FILE=my.h
```

### DSP Constraints

```bash
# Limit to 500 DSP blocks
make complete my.c my.h DSP_LIMIT=8

# Named syntax
make complete C_FILE=my.c H_FILE=my.h DSP_LIMIT=8
```

**Auto-selection logic:**

1. **No DSP_LIMIT given** → **Unconstrained**
2. **DSP_LIMIT ≥ N_param × N_solution** → **Hybrid Parallel**
3. **N_param ≤ DSP_LIMIT < N_param × N_solution** → **Solution-Level**
4. **DSP_LIMIT < N_param** → **Parameter-Level**

**Default:** `DSP_LIMIT=1000` → Unconstrained

---

## Common Commands

### Positional Syntax

| Command | Syntax | Description |
|---------|--------|-------------|
| `make complete` | `<file.c> [file.h]` | C → hardware + testbench |
| `make hardware-only` | `<config.vh>` | Config → hardware (fast) |
| `make sim_compile` | `<project_dir>` | Compile with Verilator |
| `make sim_run` | `<project_dir>` | Run simulation |
| `make sim_wave` | `<project_dir>` | Open waveforms (GTKWave) |
| `make clean` | `<project_dir>` | Remove generated files |

### Named Syntax

| Command | Parameters | Description |
|---------|------------|-------------|
| `make complete` | `C_FILE= H_FILE=` | C → hardware |
| `make hardware-only` | `CONFIG_FILE=` | Config → hardware |
| `make sim_compile` | `WORK_DIR=` | Compile simulation |
| `make list-projects` | - | Show all projects |

---

## Generated Structure

```
my_project/
├── include/          # Config (.vh, .mem)
├── benchmark/        # Test data (.txt)
├── rtl/              # Hardware (.v)
├── tb/               # Verilog testbench (.v)
├── c_tb/             # C++ testbench (.cpp)
└── obj_dir/          # Verilator build
```

---

## Architecture Examples

### Default (Unconstrained)
```bash
make complete my.c my.h
# DSP_LIMIT=1000 (default) → Unconstrained
# ✓ Full parallelism
```

### Hybrid Parallel
```bash
make complete my.c my.h DSP_LIMIT=150
# For N_param=5, N_solution=20 (needs 100 DSP)
# 150 ≥ 100 → Hybrid Parallel
# ✓ DSP for feedback, LUT for halfplanes
```

### Solution-Level
```bash
make complete my.c my.h DSP_LIMIT=50
# For N_param=5, N_solution=20
# 50 < 100 → Solution-Level
# ✓ 10 solutions in parallel
```

### Parameter-Level
```bash
make complete my.c my.h DSP_LIMIT=3
# For N_param=20
# 3 < 20 → Parameter-Level
# ✓ Minimal DSP usage
```

---

## Troubleshooting

### Config file not found
```bash
make find-config        # List all configs
make config my.c my.h   # Generate new
```

### Array name mismatch
C file must contain:
- `pdaqp_halfplanes[]`
- `pdaqp_feedbacks[]`
- `pdaqp_hp_list[]`
- `pdaqp_jump_list[]`

### Simulation errors
```bash
make clean my_project
make complete_with_sim my.c my.h
make sim_log my_project
```

---

## Examples

```bash
# PID Controller
make complete ./codegen_pid/pdaqp.c
make sim_compile ./codegen_pid
make sim_run ./codegen_pid

# Custom DSP Limit
make complete my.c my.h DSP_LIMIT=200

# Regenerate Hardware
make hardware-only my_project/include/pdaqp_config.vh

# Named syntax
make complete C_FILE=my.c H_FILE=my.h
```

---

## Parameters

### Positional
```bash
make <target> file.c        # Auto-detect as C_FILE (+ auto-find .h)
make <target> config.vh     # Auto-detect as CONFIG_FILE
make <target> my_project/   # Auto-detect as WORK_DIR
```

**Auto-detection example:**
```bash
# If both pdaqp.c and pdaqp.h exist:
make complete pdaqp.c           # ✓ Auto-finds pdaqp.h
make complete pdaqp.c custom.h  # ✓ Uses custom.h
```

### Named

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C_FILE` | - | Input C source |
| `H_FILE` | auto | Header (auto as `<basename>.h` if exists) |
| `CONFIG_FILE` | - | Verilog config (*_config.vh) |
| `DSP_LIMIT` | 1000 | Max DSP blocks |
| `WORK_DIR` | `pwd` | Project directory |

---

## Help

```bash
make help       # Full reference
make info       # Show configuration
make check      # Verify dependencies
```

---

**Verilator:** https://verilator.org  
```