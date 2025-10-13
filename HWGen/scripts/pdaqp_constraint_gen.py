#!/usr/bin/env python3
"""
PDAQP XDC Constraint Generator - Verilog Top-Level Input Version
Academic-quality constraint generation from actual hardware design
Extracts real signal names and hierarchy from Verilog source files
"""

import re
import os
import argparse
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class VerilogSignal:
    """Represents a Verilog signal with its properties"""
    name: str
    width: int
    signal_type: str  # 'input', 'output', 'wire', 'reg'
    range_spec: str   # e.g., "[31:0]"

@dataclass
class ModuleInstance:
    """Represents a module instantiation"""
    instance_name: str
    module_name: str
    connections: Dict[str, str]  # port_name -> connected_signal

@dataclass
class HardwareDesign:
    """Complete hardware design extracted from Verilog"""
    module_name: str
    ports: Dict[str, VerilogSignal]
    internal_signals: Dict[str, VerilogSignal]
    instances: List[ModuleInstance]
    parameters: Dict[str, int]
    clocks: List[str]
    resets: List[str]

@dataclass
class DSPAllocation:
    """DSP resource allocation analysis"""
    total_mults: int
    dsp_instances: List[str]
    lut_instances: List[str]
    utilization_percent: float
    strategy: str

class VerilogParser:
    """Parser for Verilog top-level files"""
    
    def __init__(self):
        # Regex patterns for Verilog parsing
        self.module_pattern = re.compile(r'module\s+(\w+)\s*(?:\#.*?)?\s*\(', re.MULTILINE)
        self.port_pattern = re.compile(r'(input|output)\s+(?:wire\s+|reg\s+)?(?:\[([^\]]+)\]\s+)?(\w+)', re.MULTILINE)
        self.signal_pattern = re.compile(r'(wire|reg)\s+(?:signed\s+)?(?:\[([^\]]+)\]\s+)?(\w+)', re.MULTILINE)
        self.instance_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(\s*(.*?)\s*\);', re.DOTALL)
        self.param_pattern = re.compile(r'localparam\s+(\w+)\s*=\s*([^;]+);')
        self.assign_pattern = re.compile(r'assign\s+(\w+)\s*=\s*([^;]+);')
        
    def parse_verilog_file(self, verilog_path: str) -> HardwareDesign:
        """Parse top-level Verilog file and extract design information"""
        
        if not os.path.exists(verilog_path):
            raise FileNotFoundError(f"Verilog file not found: {verilog_path}")
        
        with open(verilog_path, 'r') as f:
            content = f.read()
        
        # Remove comments and simplify
        content = self._preprocess_verilog(content)
        
        # Extract module name
        module_match = self.module_pattern.search(content)
        if not module_match:
            raise ValueError("No module declaration found")
        module_name = module_match.group(1)
        
        # Parse different components
        ports = self._parse_ports(content)
        internal_signals = self._parse_internal_signals(content)
        instances = self._parse_instances(content)
        parameters = self._parse_parameters(content)
        
        # Identify clocks and resets
        clocks = self._identify_clocks(ports)
        resets = self._identify_resets(ports)
        
        return HardwareDesign(
            module_name=module_name,
            ports=ports,
            internal_signals=internal_signals,
            instances=instances,
            parameters=parameters,
            clocks=clocks,
            resets=resets
        )
    
    def _preprocess_verilog(self, content: str) -> str:
        """Clean and preprocess Verilog content"""
        # Remove line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Remove block comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove timescale and includes
        content = re.sub(r'`timescale.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'`include.*$', '', content, flags=re.MULTILINE)
        return content
    
    def _parse_ports(self, content: str) -> Dict[str, VerilogSignal]:
        """Extract module ports"""
        ports = {}
        for match in self.port_pattern.finditer(content):
            direction = match.group(1)
            range_spec = match.group(2) or ""
            name = match.group(3)
            
            width = self._calculate_width(range_spec)
            ports[name] = VerilogSignal(name, width, direction, range_spec)
        
        return ports
    
    def _parse_internal_signals(self, content: str) -> Dict[str, VerilogSignal]:
        """Extract internal signals (wire/reg)"""
        signals = {}
        for match in self.signal_pattern.finditer(content):
            signal_type = match.group(1)
            range_spec = match.group(2) or ""
            name = match.group(3)
            
            width = self._calculate_width(range_spec)
            signals[name] = VerilogSignal(name, width, signal_type, range_spec)
        
        return signals
    
    def _parse_instances(self, content: str) -> List[ModuleInstance]:
        """Extract module instances"""
        instances = []
        
        # Find module instantiations
        instance_blocks = re.findall(r'(\w+)\s+(\w+)\s*\((.*?)\);', content, re.DOTALL)
        
        for module_name, instance_name, connections_str in instance_blocks:
            # Skip primitive assignments
            if module_name in ['assign', 'always', 'initial']:
                continue
                
            connections = self._parse_connections(connections_str)
            instances.append(ModuleInstance(instance_name, module_name, connections))
        
        return instances
    
    def _parse_connections(self, connections_str: str) -> Dict[str, str]:
        """Parse port connections in module instantiation"""
        connections = {}
        
        # Match .port_name(signal_name) pattern
        conn_pattern = re.compile(r'\.(\w+)\s*\(\s*([^)]+)\s*\)')
        for match in conn_pattern.finditer(connections_str):
            port_name = match.group(1)
            signal_name = match.group(2).strip()
            connections[port_name] = signal_name
        
        return connections
    
    def _parse_parameters(self, content: str) -> Dict[str, int]:
        """Extract localparam definitions"""
        parameters = {}
        for match in self.param_pattern.finditer(content):
            name = match.group(1)
            value_str = match.group(2).strip()
            
            try:
                # Handle different number formats
                if value_str.startswith('0x'):
                    value = int(value_str, 16)
                elif value_str.startswith('0b'):
                    value = int(value_str, 2)
                else:
                    value = int(value_str)
                parameters[name] = value
            except ValueError:
                # Keep as string if not convertible
                parameters[name] = value_str
        
        return parameters
    
    def _calculate_width(self, range_spec: str) -> int:
        """Calculate signal width from range specification"""
        if not range_spec:
            return 1
        
        # Parse [msb:lsb] format
        match = re.match(r'(\d+):(\d+)', range_spec)
        if match:
            msb = int(match.group(1))
            lsb = int(match.group(2))
            return abs(msb - lsb) + 1
        
        # Parse [width-1:0] format
        match = re.match(r'(\d+)-1:0', range_spec)
        if match:
            return int(match.group(1))
        
        return 1
    
    def _identify_clocks(self, ports: Dict[str, VerilogSignal]) -> List[str]:
        """Identify clock signals"""
        clocks = []
        for name, signal in ports.items():
            if 'clk' in name.lower() and signal.signal_type == 'input':
                clocks.append(name)
        return clocks
    
    def _identify_resets(self, ports: Dict[str, VerilogSignal]) -> List[str]:
        """Identify reset signals"""
        resets = []
        for name, signal in ports.items():
            if 'rst' in name.lower() and signal.signal_type == 'input':
                resets.append(name)
        return resets

class DSPAnalyzer:
    """Analyze DSP resource usage from hardware design"""
    
    def __init__(self, dsp_limit: int = 8):
        self.dsp_limit = dsp_limit
    
    def analyze_dsp_usage(self, design: HardwareDesign) -> DSPAllocation:
        """Analyze DSP resource allocation from design"""
        
        # Find BST instance for DSP analysis
        bst_instance = None
        for instance in design.instances:
            if 'bst' in instance.module_name.lower():
                bst_instance = instance
                break
        
        if not bst_instance:
            return DSPAllocation(0, [], [], 0.0, "no_dsp")
        
        # Estimate DSP requirements from BST connections
        param_count = len([conn for port, conn in bst_instance.connections.items() 
                          if 'param_in' in port])
        sol_count = len([conn for port, conn in bst_instance.connections.items() 
                        if 'sol_out' in port])
        
        total_mults = param_count * sol_count
        
        # Determine allocation strategy
        if total_mults <= self.dsp_limit:
            strategy = "full_parallel"
            dsp_instances = [f"bst_inst/*dsp_{i//sol_count}_{i%sol_count}_*" 
                           for i in range(total_mults)]
            lut_instances = [f"bst_inst/*hp_eval_*"]
            utilization = (total_mults / self.dsp_limit) * 100
        else:
            strategy = "time_division_mux"
            dsp_instances = [f"bst_inst/*feedback_dsp_{i}_*" 
                           for i in range(self.dsp_limit)]
            lut_instances = [f"bst_inst/*hp_eval_*", f"bst_inst/*overflow_mult_*"]
            utilization = 100.0
        
        return DSPAllocation(
            total_mults=total_mults,
            dsp_instances=dsp_instances,
            lut_instances=lut_instances,
            utilization_percent=utilization,
            strategy=strategy
        )

class XDCConstraintGenerator:
    """Generate XDC constraints from parsed Verilog design"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_constraints(self, design: HardwareDesign, dsp_allocation: DSPAllocation) -> str:
        """Generate complete XDC constraint file"""
        
        sections = [
            self._generate_header(design, dsp_allocation),
            self._generate_clock_constraints(design),
            self._generate_io_constraints(design),
            self._generate_axi_constraints(design),
            self._generate_dsp_constraints(design, dsp_allocation),
            self._generate_timing_constraints(design),
            self._generate_placement_constraints(design, dsp_allocation),
            self._generate_memory_constraints(design),
            self._generate_optimization_constraints(dsp_allocation)
        ]
        
        return '\n\n'.join(sections)
    
    def _generate_header(self, design: HardwareDesign, dsp_allocation: DSPAllocation) -> str:
        """Generate constraint file header"""
        axi_input_width = design.ports.get('s_axis_tdata', VerilogSignal('', 32, '', '')).width
        axi_output_width = design.ports.get('m_axis_tdata', VerilogSignal('', 80, '', '')).width
        
        return f'''# =====================================================================
# Auto-Generated XDC Constraints for PDAQP Hardware Design
# Generated: {self.timestamp}
# Source: {design.module_name}.v
# =====================================================================
# Design Configuration:
#   Module: {design.module_name}
#   AXI Input Width: {axi_input_width} bits
#   AXI Output Width: {axi_output_width} bits
#   Clock Domains: {len(design.clocks)}
#   Reset Signals: {len(design.resets)}
#
# DSP Resource Analysis:
#   Total Multiplications: {dsp_allocation.total_mults}
#   DSP Utilization: {dsp_allocation.utilization_percent:.1f}%
#   Allocation Strategy: {dsp_allocation.strategy}
# ====================================================================='''
    
    def _generate_clock_constraints(self, design: HardwareDesign) -> str:
        """Generate clock constraints"""
        constraints = '''# =====================================================================
# CLOCK CONSTRAINTS
# ====================================================================='''
        
        for clk in design.clocks:
            constraints += f'''

# Primary clock: {clk}
create_clock -period 10.000 -name {clk} -waveform {{0.000 5.000}} [get_ports {clk}]
set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets {clk}_IBUF]'''
        
        for rst in design.resets:
            constraints += f'''

# Reset timing: {rst}
set_false_path -from [get_ports {rst}]
set_property ASYNC_REG TRUE [get_cells -hierarchical *{rst}*_sync*]'''
        
        return constraints
    
    def _generate_io_constraints(self, design: HardwareDesign) -> str:
        """Generate I/O constraints based on actual ports"""
        constraints = '''# =====================================================================
# I/O TIMING CONSTRAINTS
# ====================================================================='''
        
        # Process each port
        for port_name, port_signal in design.ports.items():
            if port_signal.signal_type == 'input' and 'clk' not in port_name and 'rst' not in port_name:
                if port_signal.width > 1:
                    constraints += f'''
set_input_delay -clock clk -max 1.000 [get_ports {{{port_name}[*]}}]
set_input_delay -clock clk -min 0.500 [get_ports {{{port_name}[*]}}]'''
                else:
                    constraints += f'''
set_input_delay -clock clk -max 1.000 [get_ports {port_name}]
set_input_delay -clock clk -min 0.500 [get_ports {port_name}]'''
            
            elif port_signal.signal_type == 'output':
                if port_signal.width > 1:
                    constraints += f'''
set_output_delay -clock clk -max 1.000 [get_ports {{{port_name}[*]}}]
set_output_delay -clock clk -min 0.500 [get_ports {{{port_name}[*]}}]'''
                else:
                    constraints += f'''
set_output_delay -clock clk -max 1.000 [get_ports {port_name}]
set_output_delay -clock clk -min 0.500 [get_ports {port_name}]'''
        
        # I/O standards
        constraints += '''

# I/O Standards
set_property IOSTANDARD LVCMOS33 [get_ports *]
set_property DRIVE 12 [get_ports -filter {DIRECTION == OUT}]
set_property SLEW FAST [get_ports -filter {DIRECTION == OUT}]'''
        
        return constraints
    
    def _generate_axi_constraints(self, design: HardwareDesign) -> str:
        """Generate AXI4-Stream specific constraints"""
        
        # Detect AXI signals
        axi_signals = [name for name in design.ports.keys() if 'axis' in name]
        if not axi_signals:
            return "# No AXI4-Stream interface detected"
        
        # Calculate transfer cycles
        input_width = design.ports.get('s_axis_tdata', VerilogSignal('', 32, '', '')).width
        cycles_needed = design.parameters.get('CYCLES_NEEDED', 5)
        
        return f'''# =====================================================================
# AXI4-Stream INTERFACE CONSTRAINTS
# =====================================================================

# AXI4-Stream handshake timing
set_max_delay -datapath_only 6.000 -from [get_cells -hierarchical *state*] -to [get_ports s_axis_tready]
set_max_delay -datapath_only 6.000 -from [get_cells -hierarchical *output_valid_reg*] -to [get_ports m_axis_tvalid]

# Parameter collection multi-cycle path ({cycles_needed} cycles for {input_width}-bit AXI)
set_multicycle_path -setup {cycles_needed} -from [get_ports {{s_axis_tdata[*]}}] -to [get_cells -hierarchical *bst_start_pulse*]
set_multicycle_path -hold {cycles_needed-1} -from [get_ports {{s_axis_tdata[*]}}] -to [get_cells -hierarchical *bst_start_pulse*]

# AXI data path constraints
set_max_delay -datapath_only 8.000 -from [get_ports {{s_axis_tdata[*]}}] -to [get_cells -hierarchical *param_buffer*]
set_max_delay -datapath_only 7.000 -from [get_cells -hierarchical *output_data_reg*] -to [get_ports {{m_axis_tdata[*]}}]'''
    
    def _generate_dsp_constraints(self, design: HardwareDesign, dsp_allocation: DSPAllocation) -> str:
        """Generate DSP allocation constraints"""
        
        bst_instance = None
        for instance in design.instances:
            if 'bst' in instance.module_name.lower():
                bst_instance = instance
                break
        
        if not bst_instance:
            return "# No BST instance found for DSP constraints"
        
        instance_name = bst_instance.instance_name
        
        constraints = f'''# =====================================================================
# DSP ALLOCATION CONSTRAINTS - {dsp_allocation.strategy.upper()}
# Strategy: {dsp_allocation.strategy} ({dsp_allocation.utilization_percent:.1f}% utilization)
# =====================================================================

# Global DSP disable for precise control
set_property USE_DSP NO [get_cells -hierarchical *]

# Enable DSP for feedback computation in BST module'''
        
        for dsp_pattern in dsp_allocation.dsp_instances:
            constraints += f'''
set_property USE_DSP YES [get_cells -hierarchical {dsp_pattern}]
set_property MULT_STYLE dsp [get_cells -hierarchical {dsp_pattern}]'''
        
        constraints += '''

# Force LUT implementation for halfplane evaluation'''
        
        for lut_pattern in dsp_allocation.lut_instances:
            constraints += f'''
set_property USE_DSP NO [get_cells -hierarchical {lut_pattern}]
set_property MULT_STYLE lut [get_cells -hierarchical {lut_pattern}]'''
        
        constraints += f'''

# DSP configuration attributes
set_property DSP_A_B_DATA A_B [get_cells -hierarchical {instance_name}/*dsp*]
set_property USE_SIMD FOUR12 [get_cells -hierarchical {instance_name}/*dsp*] -quiet'''
        
        return constraints
    
    def _generate_timing_constraints(self, design: HardwareDesign) -> str:
        """Generate internal timing constraints"""
        
        # Find BST instance for timing analysis
        bst_instance = None
        for instance in design.instances:
            if 'bst' in instance.module_name.lower():
                bst_instance = instance
                break
        
        if not bst_instance:
            return "# No BST instance found for timing constraints"
        
        instance_name = bst_instance.instance_name
        
        return f'''# =====================================================================
# INTERNAL TIMING CONSTRAINTS
# =====================================================================

# Critical path 1: Parameter extraction to BST input
set_max_delay -datapath_only 5.000 -from [get_cells -hierarchical *param_*] -to [get_pins -hierarchical {instance_name}/param_in_*]

# Critical path 2: BST computation to solution output
set_max_delay -datapath_only 8.000 -from [get_pins -hierarchical {instance_name}/sol_out_*] -to [get_cells -hierarchical *output_data_reg*]

# Critical path 3: State machine transitions
set_max_delay -datapath_only 6.000 -from [get_cells -hierarchical *state*] -to [get_cells -hierarchical *next_state*]

# Critical path 4: Counter logic
set_max_delay -datapath_only 5.000 -from [get_cells -hierarchical *receive_count*] -to [get_cells -hierarchical *bst_start_pulse*]

# BST internal timing (hierarchical)
set_max_delay -datapath_only 12.000 -from [get_pins -hierarchical {instance_name}/valid_in] -to [get_pins -hierarchical {instance_name}/valid_out]'''
    
    def _generate_placement_constraints(self, design: HardwareDesign, dsp_allocation: DSPAllocation) -> str:
        """Generate placement constraints"""
        
        constraints = '''# =====================================================================
# PLACEMENT CONSTRAINTS
# =====================================================================

# Create placement regions for major components'''
        
        # DSP placement
        if dsp_allocation.dsp_instances:
            constraints += f'''

# DSP block placement ({len(dsp_allocation.dsp_instances)} instances)
create_pblock pblock_dsp_feedback
add_cells_to_pblock [get_pblocks pblock_dsp_feedback] [get_cells -hierarchical -filter {{NAME =~ "*dsp*" && IS_PRIMITIVE}}]
resize_pblock [get_pblocks pblock_dsp_feedback] -add {{DSP48E2_X0Y0:DSP48E2_X0Y{len(dsp_allocation.dsp_instances)-1}}}'''
        
        # Memory placement
        constraints += '''

# Memory placement for lookup tables
create_pblock pblock_lut_memory
add_cells_to_pblock [get_pblocks pblock_lut_memory] [get_cells -hierarchical -filter {NAME =~ "*halfplanes*" || NAME =~ "*feedbacks*"}]
resize_pblock [get_pblocks pblock_lut_memory] -add {RAMB36_X0Y0:RAMB36_X1Y3}

# State machine placement (control logic)
create_pblock pblock_control
add_cells_to_pblock [get_pblocks pblock_control] [get_cells -hierarchical -filter {NAME =~ "*state*" || NAME =~ "*count*"}]
resize_pblock [get_pblocks pblock_control] -add {SLICE_X0Y0:SLICE_X15Y49}'''
        
        return constraints
    
    def _generate_memory_constraints(self, design: HardwareDesign) -> str:
        """Generate memory and storage constraints"""
        return '''# =====================================================================
# MEMORY CONSTRAINTS
# =====================================================================

# ROM style for lookup tables
set_property ROM_STYLE distributed [get_cells -hierarchical -filter {NAME =~ "*halfplanes*"}]
set_property ROM_STYLE distributed [get_cells -hierarchical -filter {NAME =~ "*feedbacks*"}]
set_property ROM_STYLE distributed [get_cells -hierarchical -filter {NAME =~ "*hp_list*"}]

# RAM style for pipeline registers  
set_property RAM_STYLE register [get_cells -hierarchical -filter {NAME =~ "*param_buffer*"}]
set_property RAM_STYLE register [get_cells -hierarchical -filter {NAME =~ "*output_data_reg*"}]

# Memory attributes
set_property MEMORY_PRIMITIVE_MERGE FALSE [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ "BMEM.*"}]
set_property CASCADE_HEIGHT 1 [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ "BMEM.bram.*"}]'''
    
    def _generate_optimization_constraints(self, dsp_allocation: DSPAllocation) -> str:
        """Generate synthesis and implementation optimization constraints"""
        
        # Choose strategy based on DSP utilization
        if dsp_allocation.utilization_percent >= 90:
            synth_strategy = "Performance_Explore"
            impl_strategy = "Performance_ExtraTimingOpt"
        elif dsp_allocation.utilization_percent >= 75:
            synth_strategy = "Flow_PerfOptimized_high"  
            impl_strategy = "Performance_Explore"
        else:
            synth_strategy = "Flow_AreaOptimized_high"
            impl_strategy = "Area_Explore"
        
        return f'''# =====================================================================
# OPTIMIZATION CONSTRAINTS
# =====================================================================

# Synthesis strategy (based on {dsp_allocation.utilization_percent:.1f}% DSP utilization)
set_property STRATEGY {synth_strategy} [get_runs synth_1]
set_property STRATEGY {impl_strategy} [get_runs impl_1]

# Synthesis options
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING auto [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.SHREG_MIN_SIZE 5 [get_runs synth_1]

# Implementation options
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

# Hierarchy preservation for debug
set_property KEEP_HIERARCHY TRUE [get_cells -hierarchical -filter {{NAME =~ "*bst*"}}]
set_property DONT_TOUCH TRUE [get_cells -hierarchical -filter {{NAME =~ "*dsp*" && IS_PRIMITIVE}}]

# Power optimization
set_property POWER_OPT.PAR_NUM_FANOUT_OPT 1 [current_design]
set_property STEPS.SYNTH_DESIGN.ARGS.GATED_CLOCK_CONVERSION auto [get_runs synth_1]'''

def main():
    """Main script entry point"""
    parser = argparse.ArgumentParser(
        description='Generate XDC constraints from Verilog top-level design',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python pdqp_xdc_from_verilog.py design/pdaqp_top.v
  python pdqp_xdc_from_verilog.py design/pdaqp_top.v -o constraints.xdc
  python pdqp_xdc_from_verilog.py design/pdaqp_top.v --dsp-limit 16 -v
        '''
    )
    
    parser.add_argument('verilog_file', help='Verilog top-level design file (.v)')
    parser.add_argument('-o', '--output', help='Output XDC file (default: auto-generated)')
    parser.add_argument('--dsp-limit', type=int, default=8, help='DSP resource limit (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Parse Verilog design
        if args.verbose:
            print(f"Parsing Verilog design: {args.verilog_file}")
        
        verilog_parser = VerilogParser()
        design = verilog_parser.parse_verilog_file(args.verilog_file)
        
        if args.verbose:
            print(f"Module: {design.module_name}")
            print(f"Ports: {len(design.ports)} ({len([p for p in design.ports.values() if p.signal_type == 'input'])} inputs, {len([p for p in design.ports.values() if p.signal_type == 'output'])} outputs)")
            print(f"Instances: {len(design.instances)}")
            print(f"Parameters: {len(design.parameters)}")
        
        # Analyze DSP usage
        dsp_analyzer = DSPAnalyzer(args.dsp_limit)
        dsp_allocation = dsp_analyzer.analyze_dsp_usage(design)
        
        if args.verbose:
            print(f"DSP Analysis: {dsp_allocation.strategy} strategy")
            print(f"DSP Utilization: {dsp_allocation.utilization_percent:.1f}%")
            print(f"Total Multiplications: {dsp_allocation.total_mults}")
        
        # Generate XDC constraints
        generator = XDCConstraintGenerator()
        constraints = generator.generate_constraints(design, dsp_allocation)
        
        # Determine output filename
        if args.output:
            output_file = args.output
        else:
            base_name = Path(args.verilog_file).stem
            output_file = f"{base_name}_constraints.xdc"
        
        # Write XDC file
        with open(output_file, 'w') as f:
            f.write(constraints)
        
        print(f"✓ Generated XDC constraints: {output_file}")
        print(f"  Source: {design.module_name}.v")
        print(f"  Strategy: {dsp_allocation.strategy}")
        print(f"  DSP Usage: {len(dsp_allocation.dsp_instances)} blocks ({dsp_allocation.utilization_percent:.1f}%)")
        print(f"  Format: Standard Vivado XDC")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())