#!/bin/bash

set -e

# ============================================================================
# PDAQP Verilator Simulation Runner
# ============================================================================

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TOP_MODULE="${TOP_MODULE:-pdaqp_top}"

# Auto-detect project root
if [ ! -d "$PROJECT_ROOT/rtl" ]; then
    if [ -d "$(dirname "$PROJECT_ROOT")/rtl" ]; then
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    elif [ -d "$(dirname "$(dirname "$PROJECT_ROOT")")/rtl" ]; then
        PROJECT_ROOT="$(dirname "$(dirname "$PROJECT_ROOT")")"
    fi
fi

RTL_DIR="${RTL_DIR:-$PROJECT_ROOT/rtl}"
INCLUDE_DIR="${INCLUDE_DIR:-$PROJECT_ROOT/include}"
CPP_TB="${CPP_TB:-$PROJECT_ROOT/c_tb/sim_main.cpp}"
BUILD_DIR="${BUILD_DIR:-/tmp/${USER}_pdaqp_build/build}"
OBJ_DIR="${OBJ_DIR:-$BUILD_DIR/obj_dir}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

COMMAND="${1:-all}"

check_files() {
    echo -e "${CYAN}Checking required files...${NC}"
    echo "  Project root: $PROJECT_ROOT"
    
    local errors=0
    
    [ -d "$RTL_DIR" ] && echo -e "${GREEN}✓${NC} RTL dir:      $RTL_DIR" || { echo -e "${RED}✗${NC} RTL dir not found"; errors=1; }
    [ -f "$CPP_TB" ] && echo -e "${GREEN}✓${NC} sim_main.cpp: $CPP_TB" || { echo -e "${RED}✗${NC} sim_main.cpp not found"; errors=1; }
    [ -d "$INCLUDE_DIR" ] && echo -e "${GREEN}✓${NC} Include dir:  $INCLUDE_DIR" || { echo -e "${RED}✗${NC} Include dir not found"; errors=1; }
    
    RTL_COUNT=$(ls -1 "$RTL_DIR"/*.v 2>/dev/null | wc -l)
    [ "$RTL_COUNT" -gt 0 ] && echo -e "${GREEN}✓${NC} RTL files:    $RTL_COUNT file(s)" || { echo -e "${RED}✗${NC} No RTL files"; errors=1; }
    
    echo ""
    
    if [ $errors -ne 0 ]; then
        echo -e "${RED}Error: Missing required files${NC}"
        echo -e "${YELLOW}Set PROJECT_ROOT: PROJECT_ROOT=/path/to/project $0 $COMMAND${NC}"
        exit 1
    fi
}

setup_build_dir() {
    echo -e "${CYAN}═══ Setting up build environment ═══${NC}"
    mkdir -p "$BUILD_DIR" "$OBJ_DIR"
    
    echo "Copying C++ testbench..."
    cp "$CPP_TB" "$BUILD_DIR/"
    
    echo "Creating symlinks..."
    rm -f "$BUILD_DIR/include" "$OBJ_DIR/include"
    ln -sf "$INCLUDE_DIR" "$BUILD_DIR/include"
    ln -sf "$INCLUDE_DIR" "$OBJ_DIR/include"
    echo "  $BUILD_DIR/include -> $INCLUDE_DIR"
    echo "  $OBJ_DIR/include -> $INCLUDE_DIR"
    echo ""
    echo -e "${GREEN}✓ Setup complete${NC}"
    echo ""
}

compile_design() {
    echo -e "${CYAN}═══ Compiling with Verilator ═══${NC}"
    
    RTL_FILES=$(find "$RTL_DIR" -name "*.v")
    echo "RTL sources:"
    echo "$RTL_FILES" | sed 's/^/  /'
    echo ""
    
    cd "$BUILD_DIR"
    verilator -Wall -Wno-fatal --trace \
        +incdir+"$INCLUDE_DIR" \
        --cc --exe \
        --top-module "$TOP_MODULE" \
        --Mdir "$OBJ_DIR" \
        -CFLAGS "-std=c++14 -I$(dirname $CPP_TB)" \
        --build \
        -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-INCABSPATH \
        $(basename "$CPP_TB") \
        $RTL_FILES
    
    echo ""
    [ -f "$OBJ_DIR/V$TOP_MODULE" ] && echo -e "${GREEN}✓ Compilation successful${NC}" || { echo -e "${RED}✗ Compilation failed${NC}"; exit 1; }
    echo -e "${CYAN}Executable: $OBJ_DIR/V$TOP_MODULE${NC}"
}

run_simulation() {
    echo -e "${CYAN}═══ Running Simulation ═══${NC}"
    
    EXECUTABLE="$OBJ_DIR/V$TOP_MODULE"
    [ -f "$EXECUTABLE" ] || { echo -e "${RED}Executable not found. Run 'compile' first.${NC}"; exit 1; }
    
    echo "Working dir: $OBJ_DIR"
    echo ""
    
    cd "$OBJ_DIR"
    ./V$TOP_MODULE
    
    echo ""
    echo -e "${CYAN}═══ Simulation Complete ═══${NC}"
    
    VCD=$(find . -name "*.vcd" | head -1)
    [ -n "$VCD" ] && echo -e "${GREEN}✓${NC} Waveform: $OBJ_DIR/$VCD"
    
    LOG=$(find . -name "*_test_log.txt" | head -1)
    [ -n "$LOG" ] && echo -e "${GREEN}✓${NC} Log file: $OBJ_DIR/$LOG"
}

clean_build() {
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✓ Cleaned${NC}"
}

show_help() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        PDAQP Verilator Simulation Runner                      ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC} $0 {compile|simulate|all|clean|help}"
    echo ""
    echo "  compile   - Compile the design"
    echo "  simulate  - Run simulation"
    echo "  all       - Compile and simulate"
    echo "  clean     - Clean build directory"
    echo "  help      - Show this help"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  PROJECT_ROOT - Project path (default: current dir)"
    echo "  TOP_MODULE   - Top module (default: pdaqp_top)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./run_simulation.sh all"
    echo "  PROJECT_ROOT=/path/to/project ./run_simulation.sh compile"
    echo ""
}

case "$COMMAND" in
    compile)
        check_files
        setup_build_dir
        compile_design
        ;;
    simulate)
        check_files
        run_simulation
        ;;
    all)
        check_files
        setup_build_dir
        compile_design
        run_simulation
        ;;
    clean)
        clean_build
        ;;
    help|*)
        show_help
        ;;
esac