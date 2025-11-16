# PDAQP Hardware Generator Makefile with Verilator Integration

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CONFIG_FILE ?= 
DSP_LIMIT ?= 1000

# Input files
C_FILE ?= 
H_FILE ?= 

# Positional argument parsing - extract non-target arguments
KNOWN_TARGETS := all help complete config benchmark hardware hardware-only generate_tb \
                 sim_compile sim_run sim_wave sim_log sim_clean sim_cleanall \
                 complete_with_sim hardware_with_sim clean clean_sim check info \
                 find-config list-projects complete_tb_gen

ALL_GOALS := $(MAKECMDGOALS)
POS_ARGS := $(filter-out $(KNOWN_TARGETS),$(ALL_GOALS))
FIRST_ARG := $(word 1,$(POS_ARGS))
SECOND_ARG := $(word 2,$(POS_ARGS))

# Smart argument detection based on file extension or existence
ifneq ($(FIRST_ARG),)
  ifeq ($(suffix $(FIRST_ARG)),.c)
    # .c file detected
    C_FILE := $(FIRST_ARG)
    ifneq ($(SECOND_ARG),)
      H_FILE := $(SECOND_ARG)
    endif
  else ifeq ($(suffix $(FIRST_ARG)),.vh)
    # .vh config file detected
    CONFIG_FILE := $(FIRST_ARG)
  else ifneq ($(wildcard $(FIRST_ARG)),)
    # Directory detected
    WORK_DIR := $(FIRST_ARG)
  endif
endif

# Auto-detect base directory and header file
ifdef C_FILE
  BASE_DIR := $(dir $(C_FILE))
  # Auto-detect .h file if not specified
  ifndef H_FILE
    POSSIBLE_H := $(patsubst %.c,%.h,$(C_FILE))
    ifneq ($(wildcard $(POSSIBLE_H)),)
      H_FILE := $(POSSIBLE_H)
      $(info Auto-detected H_FILE: $(H_FILE))
    endif
  endif
else ifdef CONFIG_FILE
  BASE_DIR := $(dir $(CONFIG_FILE))..
else
  BASE_DIR := .
endif

# Output directories
VERILOG_DIR := $(BASE_DIR)/include
BENCHMARK_DIR := $(BASE_DIR)/benchmark
HARDWARE_DIR := $(BASE_DIR)

# Working directory for simulation
WORK_DIR ?= $(shell pwd)

# Python interpreter
PYTHON ?= python3

# Script directory (relative to Makefile location)
SCRIPT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/scripts

# Python scripts for hardware generation
SCRIPTS = $(SCRIPT_DIR)/generate_unconstrained.py \
          $(SCRIPT_DIR)/generate_hybrid_parallel.py \
          $(SCRIPT_DIR)/generate_solution_level.py \
          $(SCRIPT_DIR)/generate_parameter_level.py \
          $(SCRIPT_DIR)/auto_select_generator.py \
          $(SCRIPT_DIR)/generate_verilator_testbench.py

# Software utility scripts
SW_SCRIPTS = $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_benchmark.py \
             $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_files.py

# Verilator tools directory
VERILATOR_DIR = $(SCRIPT_DIR)/sw_tb_gen
VERILATOR_MAKEFILE = $(VERILATOR_DIR)/Makefile
VERILATOR_SCRIPT = $(VERILATOR_DIR)/run_simulation.sh

# Terminal colors
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# ═══════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════

.PHONY: all help complete config benchmark hardware hardware-only generate_tb \
        sim_compile sim_run sim_wave sim_log sim_clean sim_cleanall \
        complete_with_sim hardware_with_sim clean clean_sim check info \
        find-config list-projects complete_tb_gen

# Catch-all rule for positional arguments (treat as dummy targets)
%:
	@:

# ═══════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════

all: help

# ═══════════════════════════════════════════════════════════════════════
# HELP MESSAGE
# ═══════════════════════════════════════════════════════════════════════

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║        PDAQP Hardware Generation & Simulation Makefile         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)═══ Quick Start (Positional Syntax) ═══$(NC)"
	@echo ""
	@echo "$(YELLOW)# From C source:$(NC)"
	@echo "  make complete my_project/pdaqp.c my_project/pdaqp.h"
	@echo "  make complete my_project/pdaqp.c                      $(GREEN)# Auto-find .h$(NC)"
	@echo ""
	@echo "$(YELLOW)# From config file:$(NC)"
	@echo "  make hardware-only my_project/include/pdaqp_config.vh"
	@echo ""
	@echo "$(YELLOW)# Simulation:$(NC)"
	@echo "  make sim_compile my_project"
	@echo "  make sim_run my_project"
	@echo ""
	@echo "$(CYAN)═══ Main Workflows ═══$(NC)"
	@echo ""
	@echo "  $(GREEN)complete$(NC)              Generate everything from C source"
	@echo "                        Positional: make complete <file.c> [file.h] [DSP_LIMIT=N]"
	@echo "                        Named: C_FILE=<file.c> H_FILE=<file.h> [DSP_LIMIT=N]"
	@echo ""
	@echo "  $(GREEN)complete_with_sim$(NC)     Complete + auto simulation"
	@echo ""
	@echo "  $(GREEN)hardware-only$(NC)         Fast regeneration from config"
	@echo "                        Positional: make hardware-only <config.vh> [DSP_LIMIT=N]"
	@echo ""
	@echo "  $(GREEN)hardware_with_sim$(NC)     Hardware-only + auto simulation"
	@echo ""
	@echo "$(CYAN)═══ Individual Steps ═══$(NC)"
	@echo ""
	@echo "  $(YELLOW)config$(NC)                Generate config from C source"
	@echo "  $(YELLOW)benchmark$(NC)             Generate test data"
	@echo "  $(YELLOW)hardware$(NC)              Generate RTL from config"
	@echo "  $(YELLOW)generate_tb$(NC)           Generate C++ testbench"
	@echo ""
	@echo "$(CYAN)═══ Simulation ═══$(NC)"
	@echo ""
	@echo "  $(YELLOW)sim_compile$(NC)           Compile with Verilator"
	@echo "  $(YELLOW)sim_run$(NC)               Run simulation"
	@echo "  $(YELLOW)sim_wave$(NC)              Open waveform viewer (GTKWave)"
	@echo "  $(YELLOW)sim_log$(NC)               Show simulation log"
	@echo ""
	@echo "$(CYAN)═══ Utilities ═══$(NC)"
	@echo ""
	@echo "  $(YELLOW)info$(NC)                  Show configuration"
	@echo "  $(YELLOW)check$(NC)                 Verify dependencies"
	@echo "  $(YELLOW)list-projects$(NC)         List all projects"
	@echo "  $(YELLOW)find-config$(NC)           List config files"
	@echo "  $(YELLOW)clean$(NC)                 Remove generated files"
	@echo "  $(YELLOW)clean_sim$(NC)             Remove simulation artifacts"
	@echo ""
	@echo "$(CYAN)═══ Examples ═══$(NC)"
	@echo ""
	@echo "  $(GREEN)# Complete workflow:$(NC)"
	@echo "  make complete_with_sim my.c my.h"
	@echo ""
	@echo "  $(GREEN)# Regenerate hardware:$(NC)"
	@echo "  make hardware-only my_project/include/pdaqp_config.vh"
	@echo ""
	@echo "  $(GREEN)# Simulation:$(NC)"
	@echo "  make sim_compile my_project"
	@echo "  make sim_run my_project"
	@echo ""
	@echo "  $(GREEN)# With DSP limit:$(NC)"
	@echo "  make complete my.c DSP_LIMIT=500"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════
# INTERNAL: Hardware + Testbench Generation
# Used by both complete and hardware-only workflows
# ═══════════════════════════════════════════════════════════════════════

.PHONY: complete_tb_gen
complete_tb_gen:
	@# Step 1: Generate hardware using auto-selection
	@echo "$(YELLOW)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║ Auto-Selecting Architecture & Generating Hardware             ║$(NC)"
	@echo "$(YELLOW)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@$(PYTHON) $(SCRIPT_DIR)/auto_select_generator.py $(CONFIG_FILE) \
		-o $(HARDWARE_DIR) \
		-d $(DSP_LIMIT) \
		-b $(BENCHMARK_DIR)
	@echo "$(GREEN)✓ Hardware generated$(NC)"
	@echo ""
	@# Step 2: Generate C++ testbench for Verilator
	@echo "$(YELLOW)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║ Generating C++ Testbench (sim_main.cpp)                       ║$(NC)"
	@echo "$(YELLOW)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@CONFIG_DIR=$$(dirname $(CONFIG_FILE)); \
	mkdir -p $(HARDWARE_DIR)/c_tb; \
	$(PYTHON) $(SCRIPT_DIR)/generate_verilator_testbench.py $(CONFIG_FILE) \
		-i $$CONFIG_DIR \
		-o $(HARDWARE_DIR)/c_tb
	@echo "$(GREEN)✓ C++ testbench generated$(NC)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════
# COMPLETE WORKFLOW: C → config → benchmark → hardware → testbench
# ═══════════════════════════════════════════════════════════════════════

complete:
	@# Validate inputs
	@test -n "$(C_FILE)" || { \
		echo "$(RED)Error: C_FILE required$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage (positional):$(NC)"; \
		echo "  make complete <file.c> [file.h]"; \
		echo ""; \
		echo "$(YELLOW)Usage (named):$(NC)"; \
		echo "  make complete C_FILE=<file.c> H_FILE=<file.h>"; \
		echo ""; \
		exit 1; \
	}
	@test -n "$(H_FILE)" || { \
		echo "$(RED)Error: H_FILE required$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage:$(NC)"; \
		echo "  make complete $(C_FILE) <file.h>"; \
		echo "  OR"; \
		echo "  make complete C_FILE=$(C_FILE) H_FILE=<file.h>"; \
		echo ""; \
		exit 1; \
	}
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║           PDAQP Complete Hardware Generation                   ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Input files:$(NC)"
	@echo "  C file:  $(C_FILE)"
	@echo "  H file:  $(H_FILE)"
	@echo "  DSP:     $(DSP_LIMIT)"
	@echo ""
	@# Step 1: Generate Verilog config
	@echo "$(YELLOW)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║ [1/4] Generating Verilog Configuration                        ║$(NC)"
	@echo "$(YELLOW)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@mkdir -p $(VERILOG_DIR)
	@$(PYTHON) $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_files.py -c $(C_FILE) -H $(H_FILE) -o $(VERILOG_DIR)
	@echo "$(GREEN)✓ Config generated$(NC)"
	@echo ""
	@# Step 2: Generate benchmark
	@echo "$(YELLOW)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║ [2/4] Generating Benchmark Test Data                          ║$(NC)"
	@echo "$(YELLOW)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@mkdir -p $(BENCHMARK_DIR)
	@$(PYTHON) $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_benchmark.py -c $(C_FILE) -H $(H_FILE) -o $(BENCHMARK_DIR)
	@echo "$(GREEN)✓ Benchmark generated$(NC)"
	@echo ""
	@# Step 3-4: Delegate to complete_tb_gen
	@BASE_NAME=$$(basename $(C_FILE) .c); \
	GENERATED_CONFIG=$(VERILOG_DIR)/$${BASE_NAME}_config.vh; \
	$(MAKE) complete_tb_gen CONFIG_FILE=$$GENERATED_CONFIG HARDWARE_DIR=$(HARDWARE_DIR) BENCHMARK_DIR=$(BENCHMARK_DIR) DSP_LIMIT=$(DSP_LIMIT)
	@# Summary
	@echo "$(YELLOW)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║ Complete Workflow Summary                                     ║$(NC)"
	@echo "$(YELLOW)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)✓ Workflow complete$(NC)"
	@echo ""
	@echo "$(CYAN)Generated:$(NC)"
	@echo "  $(HARDWARE_DIR)/include/    $(GREEN)✓$(NC) Config (.vh, .mem)"
	@echo "  $(HARDWARE_DIR)/benchmark/  $(GREEN)✓$(NC) Test data (.txt)"
	@echo "  $(HARDWARE_DIR)/rtl/        $(GREEN)✓$(NC) Hardware (.v)"
	@echo "  $(HARDWARE_DIR)/tb/         $(GREEN)✓$(NC) Verilog testbench (.v)"
	@echo "  $(HARDWARE_DIR)/c_tb/       $(GREEN)✓$(NC) C++ testbench (.cpp)"
	@echo ""
	@echo "$(MAGENTA)Next:$(NC) make sim_compile $(HARDWARE_DIR)"
	@echo ""

# Complete + auto simulation
complete_with_sim: complete
	@echo ""
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║              Compiling with Verilator                         ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) sim_compile WORK_DIR=$(HARDWARE_DIR)
	@echo ""
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║                   Running Simulation                          ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) sim_run WORK_DIR=$(HARDWARE_DIR)
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║     ✓ Complete + Simulation Finished                          ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════
# HARDWARE-ONLY WORKFLOW: config → hardware → testbench (fast path)
# ═══════════════════════════════════════════════════════════════════════

hardware-only:
	@# Validate config file
	@test -n "$(CONFIG_FILE)" || { \
		echo "$(RED)Error: CONFIG_FILE required$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage (positional):$(NC)"; \
		echo "  make hardware-only <config.vh>"; \
		echo ""; \
		echo "$(YELLOW)Usage (named):$(NC)"; \
		echo "  make hardware-only CONFIG_FILE=<config.vh>"; \
		echo ""; \
		echo "$(MAGENTA)Tip: make find-config$(NC)"; \
		echo ""; \
		exit 1; \
	}
	@test -f "$(CONFIG_FILE)" || { echo "$(RED)Error: File not found: $(CONFIG_FILE)$(NC)"; exit 1; }
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║           PDAQP Hardware-Only Generation (Fast)                ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Config:$(NC) $(CONFIG_FILE)"
	@echo "$(CYAN)Output:$(NC) $(HARDWARE_DIR)"
	@echo "$(CYAN)DSP:$(NC)    $(DSP_LIMIT)"
	@echo ""
	@# Delegate to complete_tb_gen
	@$(MAKE) complete_tb_gen CONFIG_FILE=$(CONFIG_FILE) HARDWARE_DIR=$(HARDWARE_DIR) BENCHMARK_DIR=$(BENCHMARK_DIR) DSP_LIMIT=$(DSP_LIMIT)
	@echo "$(GREEN)✓ Hardware-only complete$(NC)"
	@echo ""
	@echo "$(CYAN)Generated:$(NC)"
	@echo "  $(HARDWARE_DIR)/rtl/   $(GREEN)✓$(NC) Hardware (.v)"
	@echo "  $(HARDWARE_DIR)/tb/    $(GREEN)✓$(NC) Verilog testbench (.v)"
	@echo "  $(HARDWARE_DIR)/c_tb/  $(GREEN)✓$(NC) C++ testbench (.cpp)"
	@echo ""
	@echo "$(MAGENTA)Next:$(NC) make sim_compile $(HARDWARE_DIR)"
	@echo ""

# Hardware-only + auto simulation
hardware_with_sim: hardware-only
	@echo ""
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║              Compiling with Verilator                         ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) sim_compile WORK_DIR=$(HARDWARE_DIR)
	@echo ""
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║                   Running Simulation                          ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) sim_run WORK_DIR=$(HARDWARE_DIR)
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║     ✓ Hardware + Simulation Finished                          ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════
# INDIVIDUAL STEPS
# ═══════════════════════════════════════════════════════════════════════

config:
	@# Generate Verilog config from C source
	@test -n "$(C_FILE)" || { echo "$(RED)Error: C_FILE required$(NC)"; echo "$(YELLOW)Usage: make config <file.c> [file.h]$(NC)"; exit 1; }
	@test -n "$(H_FILE)" || { echo "$(RED)Error: H_FILE required$(NC)"; echo "$(YELLOW)Usage: make config $(C_FILE) <file.h>$(NC)"; exit 1; }
	@echo "$(BLUE)Generating config...$(NC)"
	@mkdir -p $(VERILOG_DIR)
	@$(PYTHON) $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_files.py -c $(C_FILE) -H $(H_FILE) -o $(VERILOG_DIR)
	@echo "$(GREEN)✓ Config: $(VERILOG_DIR)$(NC)"

benchmark:
	@# Generate benchmark test data
	@test -n "$(C_FILE)" || { echo "$(RED)Error: C_FILE required$(NC)"; echo "$(YELLOW)Usage: make benchmark <file.c> [file.h]$(NC)"; exit 1; }
	@test -n "$(H_FILE)" || { echo "$(RED)Error: H_FILE required$(NC)"; echo "$(YELLOW)Usage: make benchmark $(C_FILE) <file.h>$(NC)"; exit 1; }
	@echo "$(BLUE)Generating benchmark...$(NC)"
	@mkdir -p $(BENCHMARK_DIR)
	@$(PYTHON) $(SCRIPT_DIR)/sw_scripts/generate_pdaqp_benchmark.py -c $(C_FILE) -H $(H_FILE) -o $(BENCHMARK_DIR)
	@echo "$(GREEN)✓ Benchmark: $(BENCHMARK_DIR)$(NC)"

hardware:
	@# Generate hardware RTL from config
	@test -n "$(CONFIG_FILE)" || { echo "$(RED)Error: CONFIG_FILE required$(NC)"; echo "$(YELLOW)Usage: make hardware <config.vh>$(NC)"; exit 1; }
	@test -f "$(CONFIG_FILE)" || { echo "$(RED)Error: File not found: $(CONFIG_FILE)$(NC)"; exit 1; }
	@echo "$(BLUE)Generating hardware...$(NC)"
	@$(PYTHON) $(SCRIPT_DIR)/auto_select_generator.py $(CONFIG_FILE) \
		-o $(HARDWARE_DIR) \
		-d $(DSP_LIMIT) \
		-b $(BENCHMARK_DIR)
	@echo "$(GREEN)✓ Hardware generated$(NC)"

generate_tb:
	@# Generate C++ testbench for Verilator
	@test -n "$(CONFIG_FILE)" || { \
		echo "$(RED)Error: CONFIG_FILE required$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage (positional):$(NC)"; \
		echo "  make generate_tb <config.vh>"; \
		echo ""; \
		echo "$(YELLOW)Usage (named):$(NC)"; \
		echo "  make generate_tb CONFIG_FILE=<config.vh>"; \
		echo ""; \
		echo "$(MAGENTA)Tip: make find-config$(NC)"; \
		echo ""; \
		exit 1; \
	}
	@test -f "$(CONFIG_FILE)" || { echo "$(RED)Error: File not found: $(CONFIG_FILE)$(NC)"; exit 1; }
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Generating C++ Testbench (sim_main.cpp)               ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@CONFIG_DIR=$$(dirname $(CONFIG_FILE)); \
	echo "$(CYAN)Config:  $(CONFIG_FILE)$(NC)"; \
	echo "$(CYAN)Include: $$CONFIG_DIR$(NC)"; \
	echo "$(CYAN)Output:  $(HARDWARE_DIR)/c_tb$(NC)"; \
	echo ""; \
	mkdir -p $(HARDWARE_DIR)/c_tb; \
	$(PYTHON) $(SCRIPT_DIR)/generate_verilator_testbench.py $(CONFIG_FILE) \
		-i $$CONFIG_DIR \
		-o $(HARDWARE_DIR)/c_tb
	@echo ""
	@echo "$(GREEN)✓ Testbench: $(HARDWARE_DIR)/c_tb/sim_main.cpp$(NC)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════
# VERILATOR SIMULATION (portable, uses -C to change directory)
# ═══════════════════════════════════════════════════════════════════════

sim_compile:
	@# Compile with Verilator
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              Compiling with Verilator                         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Project: $(WORK_DIR)$(NC)"
	@test -f "$(VERILATOR_MAKEFILE)" || { echo "$(RED)Error: Verilator Makefile not found$(NC)"; exit 1; }
	$(MAKE) -C $(VERILATOR_DIR) compile PROJECT_ROOT=$(realpath $(WORK_DIR))
	@echo ""

sim_run:
	@# Run Verilator simulation
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                   Running Simulation                          ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Project: $(WORK_DIR)$(NC)"
	@test -f "$(VERILATOR_MAKEFILE)" || { echo "$(RED)Error: Verilator Makefile not found$(NC)"; exit 1; }
	$(MAKE) -C $(VERILATOR_DIR) simulate PROJECT_ROOT=$(realpath $(WORK_DIR))
	@echo ""

sim_wave:
	@# Open waveform viewer (GTKWave)
	$(MAKE) -C $(VERILATOR_DIR) wave PROJECT_ROOT=$(realpath $(WORK_DIR))

sim_log:
	@# Display simulation log
	$(MAKE) -C $(VERILATOR_DIR) log PROJECT_ROOT=$(realpath $(WORK_DIR))

sim_clean:
	@# Clean Verilator build directory
	@echo "$(YELLOW)Cleaning Verilator build...$(NC)"
	$(MAKE) -C $(VERILATOR_DIR) clean PROJECT_ROOT=$(realpath $(WORK_DIR)) 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

sim_cleanall:
	@# Clean all Verilator builds
	@echo "$(YELLOW)Cleaning all builds...$(NC)"
	$(MAKE) -C $(VERILATOR_DIR) cleanall PROJECT_ROOT=$(realpath $(WORK_DIR)) 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

list-projects:
	@# List all available PDAQP projects
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              Available PDAQP Projects                          ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@PROJECTS=$$(find . -maxdepth 3 -type d -name "include" 2>/dev/null | \
		while read dir; do \
			if ls "$$dir"/*_config.vh >/dev/null 2>&1; then \
				dirname "$$dir"; \
			fi; \
		done | sort -u); \
	if [ -z "$$PROJECTS" ]; then \
		echo "$(RED)No projects found$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Tip: make complete <file.c>$(NC)"; \
	else \
		COUNT=$$(echo "$$PROJECTS" | wc -l | tr -d ' '); \
		echo "$(GREEN)Found $$COUNT project(s):$(NC)"; \
		echo ""; \
		echo "$$PROJECTS" | nl -w2 -s'. ' -v1; \
		echo ""; \
		echo "$(CYAN)Structure:$(NC)"; \
		for proj in $$PROJECTS; do \
			echo ""; \
			echo "$(YELLOW)$$proj/$(NC)"; \
			[ -d "$$proj/include" ] && echo "  ├── $(GREEN)✓$(NC) include/" || echo "  ├── $(RED)✗$(NC) include/"; \
			[ -d "$$proj/benchmark" ] && echo "  ├── $(GREEN)✓$(NC) benchmark/" || echo "  ├── $(YELLOW)⚠$(NC) benchmark/"; \
			[ -d "$$proj/rtl" ] && echo "  ├── $(GREEN)✓$(NC) rtl/" || echo "  ├── $(YELLOW)⚠$(NC) rtl/"; \
			[ -d "$$proj/tb" ] && echo "  ├── $(GREEN)✓$(NC) tb/" || echo "  ├── $(YELLOW)⚠$(NC) tb/"; \
			[ -d "$$proj/c_tb" ] && echo "  ├── $(GREEN)✓$(NC) c_tb/" || echo "  ├── $(YELLOW)⚠$(NC) c_tb/"; \
			[ -d "$$proj/obj_dir" ] && echo "  └── $(CYAN)○$(NC) obj_dir/" || echo "  └── $(YELLOW)○$(NC) obj_dir/"; \
		done; \
		echo ""; \
		echo "$(CYAN)Usage:$(NC)"; \
		FIRST=$$(echo "$$PROJECTS" | head -1); \
		echo "  $(YELLOW)make sim_compile $$FIRST$(NC)"; \
		echo "  $(YELLOW)make clean $$FIRST$(NC)"; \
		echo ""; \
	fi

info:
	@# Display current configuration
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              Current Configuration                             ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Paths:$(NC)"
	@echo "  Scripts:    $(SCRIPT_DIR)"
	@echo "  Verilator:  $(VERILATOR_DIR)"
	@echo "  Work dir:   $(WORK_DIR)"
	@echo "  Base dir:   $(BASE_DIR)"
	@echo ""
	@echo "$(CYAN)Arguments:$(NC)"
	@test -n "$(C_FILE)" && echo "  C file:     $(C_FILE)" || echo "  C file:     $(YELLOW)(not set)$(NC)"
	@test -n "$(H_FILE)" && echo "  H file:     $(H_FILE)" || echo "  H file:     $(YELLOW)(not set)$(NC)"
	@test -n "$(CONFIG_FILE)" && echo "  Config:     $(CONFIG_FILE)" || echo "  Config:     $(YELLOW)(not set)$(NC)"
	@echo "  DSP limit:  $(DSP_LIMIT)"
	@echo ""
	@echo "$(CYAN)Tools:$(NC)"
	@which $(PYTHON) > /dev/null && echo "  Python:     $(GREEN)✓$(NC) $$($(PYTHON) --version)" || echo "  Python:     $(RED)✗$(NC)"
	@which verilator > /dev/null && echo "  Verilator:  $(GREEN)✓$(NC) $$(verilator --version | head -1)" || echo "  Verilator:  $(YELLOW)⚠$(NC)"
	@which gtkwave > /dev/null && echo "  GTKWave:    $(GREEN)✓$(NC)" || echo "  GTKWave:    $(YELLOW)⚠$(NC)"
	@echo ""

check:
	@# Verify all dependencies
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              Dependency Check                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(CYAN)Core Tools:$(NC)"
	@which $(PYTHON) > /dev/null || { echo "$(RED)✗ python3 not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Python3:$(NC) $$($(PYTHON) --version)"
	@which verilator > /dev/null && echo "$(GREEN)✓ Verilator:$(NC) $$(verilator --version | head -1)" || echo "$(YELLOW)⚠ Verilator not found (optional)$(NC)"
	@which gtkwave > /dev/null && echo "$(GREEN)✓ GTKWave:$(NC) installed" || echo "$(YELLOW)⚠ GTKWave not found (optional)$(NC)"
	@echo ""
	@echo "$(CYAN)Hardware Scripts:$(NC)"
	@for script in $(SCRIPTS); do \
		if [ -f $$script ]; then \
			echo "$(GREEN)✓$(NC) $$(basename $$script)"; \
		else \
			echo "$(RED)✗$(NC) $$(basename $$script)"; \
		fi; \
	done
	@echo ""
	@echo "$(CYAN)Verilator Tools:$(NC)"
	@test -f "$(VERILATOR_MAKEFILE)" && echo "$(GREEN)✓$(NC) Makefile" || echo "$(RED)✗$(NC) Makefile"
	@test -f "$(VERILATOR_SCRIPT)" && echo "$(GREEN)✓$(NC) Simulation script" || echo "$(YELLOW)⚠$(NC) Script"
	@echo ""

find-config:
	@# Search for all config files
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║           Configuration Files                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@CONFIGS=$$(find . -name "*_config.vh" 2>/dev/null | sort); \
	if [ -z "$$CONFIGS" ]; then \
		echo "$(RED)No config files found$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Tip: make complete <file.c>$(NC)"; \
		exit 1; \
	fi; \
	COUNT=$$(echo "$$CONFIGS" | wc -l); \
	echo "$(GREEN)Found $$COUNT file(s):$(NC)"; \
	echo ""; \
	echo "$$CONFIGS" | nl -w2 -s'. '; \
	echo ""; \
	echo "$(CYAN)Usage:$(NC)"; \
	FIRST=$$(echo "$$CONFIGS" | head -1); \
	echo "  $(YELLOW)make generate_tb $$FIRST$(NC)"; \
	echo "  $(YELLOW)make hardware-only $$FIRST$(NC)"; \
	echo ""

clean:
	@# Remove all generated files from project
	@test -n "$(WORK_DIR)" || { \
		echo "$(RED)╔════════════════════════════════════════════════════════════════╗$(NC)"; \
		echo "$(RED)║                    ERROR: Directory Required                   ║$(NC)"; \
		echo "$(RED)╚════════════════════════════════════════════════════════════════╝$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage: make clean <project_path>$(NC)"; \
		echo ""; \
		echo "$(MAGENTA)Available:$(NC)"; \
		$(MAKE) --no-print-directory list-projects | grep -A 999 "Found.*project"; \
		echo ""; \
		exit 1; \
	}
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║              Cleaning: $(WORK_DIR)                             ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@REAL_DIR=$$(realpath $(WORK_DIR) 2>/dev/null || echo "$(WORK_DIR)"); \
	if [ ! -d "$$REAL_DIR" ]; then \
		echo "$(RED)Error: Not found: $(WORK_DIR)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(CYAN)Removing: $$REAL_DIR$(NC)"; \
	echo ""; \
	rm -rf "$$REAL_DIR/rtl" && echo "  $(GREEN)✓$(NC) rtl/" || echo "  $(YELLOW)⚠$(NC) rtl/"; \
	rm -rf "$$REAL_DIR/tb" && echo "  $(GREEN)✓$(NC) tb/" || echo "  $(YELLOW)⚠$(NC) tb/"; \
	rm -rf "$$REAL_DIR/c_tb" && echo "  $(GREEN)✓$(NC) c_tb/" || echo "  $(YELLOW)⚠$(NC) c_tb/"; \
	rm -rf "$$REAL_DIR/obj_dir" && echo "  $(GREEN)✓$(NC) obj_dir/" || echo "  $(YELLOW)⚠$(NC) obj_dir/"; \
	rm -rf "$$REAL_DIR/include" && echo "  $(GREEN)✓$(NC) include/" || echo "  $(YELLOW)⚠$(NC) include/"; \
	rm -rf "$$REAL_DIR/benchmark" && echo "  $(GREEN)✓$(NC) benchmark/" || echo "  $(YELLOW)⚠$(NC) benchmark/"; \
	rm -f "$$REAL_DIR"/*.vcd && echo "  $(GREEN)✓$(NC) *.vcd" || true; \
	rm -f "$$REAL_DIR"/*.vvp && echo "  $(GREEN)✓$(NC) *.vvp" || true; \
	rm -f "$$REAL_DIR"/*_test_log.txt && echo "  $(GREEN)✓$(NC) logs" || true; \
	rm -f "$$REAL_DIR"/*_summary.txt && echo "  $(GREEN)✓$(NC) summaries" || true; \
	$(MAKE) -C $(VERILATOR_DIR) cleanall PROJECT_ROOT=$$REAL_DIR 2>/dev/null || true; \
	echo ""; \
	echo "$(GREEN)✓ Clean complete$(NC)"; \
	echo ""

clean_sim:
	@# Remove only simulation artifacts
	@test -n "$(WORK_DIR)" || { \
		echo "$(RED)╔════════════════════════════════════════════════════════════════╗$(NC)"; \
		echo "$(RED)║                    ERROR: Directory Required                   ║$(NC)"; \
		echo "$(RED)╚════════════════════════════════════════════════════════════════╝$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Usage: make clean_sim <project_path>$(NC)"; \
		echo ""; \
		exit 1; \
	}
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║        Cleaning Simulation: $(WORK_DIR)                       ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@REAL_DIR=$$(realpath $(WORK_DIR) 2>/dev/null || echo "$(WORK_DIR)"); \
	if [ ! -d "$$REAL_DIR" ]; then \
		echo "$(RED)Error: Not found: $(WORK_DIR)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(CYAN)Removing: $$REAL_DIR$(NC)"; \
	echo ""; \
	rm -rf "$$REAL_DIR/obj_dir" && echo "  $(GREEN)✓$(NC) obj_dir/" || echo "  $(YELLOW)⚠$(NC) obj_dir/"; \
	rm -f "$$REAL_DIR"/*.vcd && echo "  $(GREEN)✓$(NC) *.vcd" || true; \
	rm -f "$$REAL_DIR"/*_test_log.txt && echo "  $(GREEN)✓$(NC) logs" || true; \
	$(MAKE) -C $(VERILATOR_DIR) cleanall PROJECT_ROOT=$$REAL_DIR 2>/dev/null || true; \
	echo ""; \
	echo "$(GREEN)✓ Simulation cleaned$(NC)"; \
	echo ""