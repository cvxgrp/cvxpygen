# ============================================================================
# PDAQP Verilator Build System - Flexible Project Path Support
# ============================================================================

# Project configuration
PROJECT ?= pdaqp_top

# Project root - can be passed as parameter, otherwise auto-detect
PROJECT_ROOT ?= $(shell pwd)

# Auto-detect if we're in a subdirectory
ifeq ($(wildcard $(PROJECT_ROOT)/rtl),)
    ifneq ($(wildcard $(PROJECT_ROOT)/../rtl),)
        PROJECT_ROOT := $(realpath $(PROJECT_ROOT)/..)
    else ifneq ($(wildcard $(PROJECT_ROOT)/../../rtl),)
        PROJECT_ROOT := $(realpath $(PROJECT_ROOT)/../..)
    endif
endif

# Directories relative to project root
RTL_DIR = $(PROJECT_ROOT)/rtl
C_TB_DIR = $(PROJECT_ROOT)/c_tb
INC_DIR = $(PROJECT_ROOT)/include

# Build directory - using /tmp to avoid permission issues
USER_NAME := $(shell whoami)
BUILD_ROOT = /tmp/$(USER_NAME)_pdaqp_build
BUILD_DIR = $(BUILD_ROOT)/build
OBJ_DIR = $(BUILD_DIR)/obj_dir

# Get absolute paths
ABS_PROJECT_ROOT := $(realpath $(PROJECT_ROOT))
ABS_RTL_DIR := $(ABS_PROJECT_ROOT)/rtl
ABS_C_TB_DIR := $(ABS_PROJECT_ROOT)/c_tb
ABS_INC_DIR := $(ABS_PROJECT_ROOT)/include

# Source files
RTL_SOURCES := $(wildcard $(ABS_RTL_DIR)/*.v)
C_SOURCES := $(ABS_C_TB_DIR)/sim_main.cpp

# Verilator configuration
VERILATOR = verilator
VFLAGS = -Wall -Wno-fatal
VFLAGS += --trace
VFLAGS += +incdir+$(ABS_INC_DIR)
VFLAGS += --cc --exe
VFLAGS += --top-module $(PROJECT)
VFLAGS += --Mdir $(OBJ_DIR)
VFLAGS += -CFLAGS "-std=c++14 -I$(ABS_C_TB_DIR)"
VFLAGS += --build
VFLAGS += -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-INCABSPATH

# Executable
EXECUTABLE = $(OBJ_DIR)/V$(PROJECT)

# Colors
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[1;33m
CYAN := \033[0;36m
NC := \033[0m

.PHONY: all compile simulate wave log clean cleanall help check_files

all: compile

check_files:
	@echo "$(CYAN)Checking required files...$(NC)"
	@echo "  Project root:  $(ABS_PROJECT_ROOT)"
	@test -d "$(ABS_RTL_DIR)" || { echo "$(RED)✗ RTL directory not found: $(ABS_RTL_DIR)$(NC)"; exit 1; }
	@echo "$(GREEN)✓$(NC) RTL dir:        $(ABS_RTL_DIR)"
	@test -d "$(ABS_C_TB_DIR)" || { echo "$(RED)✗ C++ TB directory not found: $(ABS_C_TB_DIR)$(NC)"; exit 1; }
	@echo "$(GREEN)✓$(NC) C++ TB dir:     $(ABS_C_TB_DIR)"
	@test -f "$(C_SOURCES)" || { echo "$(RED)✗ sim_main.cpp not found: $(C_SOURCES)$(NC)"; exit 1; }
	@echo "$(GREEN)✓$(NC) sim_main.cpp:  $(C_SOURCES)"
	@test -d "$(ABS_INC_DIR)" || { echo "$(RED)✗ Include directory not found: $(ABS_INC_DIR)$(NC)"; exit 1; }
	@echo "$(GREEN)✓$(NC) Include dir:   $(ABS_INC_DIR)"
	@RTL_COUNT=$$(ls -1 $(ABS_RTL_DIR)/*.v 2>/dev/null | wc -l); \
	test $$RTL_COUNT -gt 0 || { echo "$(RED)✗ No RTL files found in $(ABS_RTL_DIR)$(NC)"; exit 1; }; \
	echo "$(GREEN)✓$(NC) RTL files:     $$RTL_COUNT file(s)"
	@echo ""

setup: check_files
	@echo "$(CYAN)═══ Setting up build environment ═══$(NC)"
	@mkdir -p $(BUILD_DIR) $(OBJ_DIR)
	@chmod -R 755 $(BUILD_ROOT)
	@echo "Build directory: $(BUILD_DIR)"
	@echo ""
	@echo "Copying C++ testbench..."
	@cp $(C_SOURCES) $(BUILD_DIR)/
	@echo ""
	@echo "Creating symlinks for runtime data..."
	@rm -f $(BUILD_DIR)/include $(OBJ_DIR)/include
	@ln -sf $(ABS_INC_DIR) $(BUILD_DIR)/include
	@ln -sf $(ABS_INC_DIR) $(OBJ_DIR)/include
	@echo "  $(BUILD_DIR)/include -> $(ABS_INC_DIR)"
	@echo "  $(OBJ_DIR)/include -> $(ABS_INC_DIR)"
	@echo ""
	@echo "$(GREEN)✓ Setup complete$(NC)"
	@echo ""

compile: setup
	@echo "$(CYAN)═══ Compiling with Verilator ═══$(NC)"
	@echo "RTL sources:"
	@ls -1 $(ABS_RTL_DIR)/*.v | sed 's/^/  /'
	@echo ""
	@cd $(BUILD_DIR) && $(VERILATOR) $(VFLAGS) sim_main.cpp $(RTL_SOURCES)
	@echo ""
	@test -f $(EXECUTABLE) || { echo "$(RED)✗ Compilation failed$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Compilation successful$(NC)"
	@chmod +x $(EXECUTABLE)
	@echo "$(CYAN)Executable: $(EXECUTABLE)$(NC)"
	@echo ""

simulate:
	@echo "$(CYAN)═══ Running Simulation ═══$(NC)"
	@test -f $(EXECUTABLE) || { echo "$(RED)Error: Executable not found$(NC)"; echo "$(YELLOW)Run 'make compile' first$(NC)"; exit 1; }
	@echo "Working directory: $(OBJ_DIR)"
	@echo "Include path:      $(OBJ_DIR)/include -> $(ABS_INC_DIR)"
	@echo ""
	@cd $(OBJ_DIR) && ./V$(PROJECT)
	@echo ""
	@echo "$(CYAN)═══ Simulation Complete ═══$(NC)"
	@test -f $(OBJ_DIR)/$(PROJECT)_sim.vcd && echo "$(GREEN)✓$(NC) Waveform: $(OBJ_DIR)/$(PROJECT)_sim.vcd" || true
	@test -f $(OBJ_DIR)/$(PROJECT)_test_log.txt && echo "$(GREEN)✓$(NC) Log file: $(OBJ_DIR)/$(PROJECT)_test_log.txt" || true
	@echo ""

wave:
	@VCD=$$(find $(OBJ_DIR) -name "*.vcd" 2>/dev/null | head -1); \
	test -n "$$VCD" && { echo "$(CYAN)Opening $$VCD$(NC)"; gtkwave $$VCD & } || \
	{ echo "$(RED)No waveform found. Run 'make simulate' first.$(NC)"; exit 1; }

log:
	@LOG=$$(find $(OBJ_DIR) -name "*_test_log.txt" 2>/dev/null | head -1); \
	test -n "$$LOG" && cat $$LOG || \
	{ echo "$(RED)No log found. Run 'make simulate' first.$(NC)"; exit 1; }

clean:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Clean complete$(NC)"

cleanall:
	@echo "$(YELLOW)Cleaning all builds...$(NC)"
	@rm -rf $(BUILD_ROOT)
	@echo "$(GREEN)✓ All builds cleaned$(NC)"

help:
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║        PDAQP Verilator Build System                           ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Targets:$(NC)"
	@echo "  compile        - Compile RTL with Verilator"
	@echo "  simulate       - Run simulation"
	@echo "  wave           - View waveform in GTKWave"
	@echo "  log            - Display simulation log"
	@echo "  clean          - Remove build artifacts"
	@echo "  cleanall       - Remove all builds"
	@echo "  check_files    - Verify required files"
	@echo "  help           - Show this help"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make compile                                    # Use current dir"
	@echo "  make compile PROJECT_ROOT=/path/to/project     # Specify path"
	@echo ""
	@echo "$(CYAN)Current Config:$(NC)"
	@echo "  Project root: $(ABS_PROJECT_ROOT)"
	@echo "  Build dir:    $(BUILD_DIR)"
	@echo ""