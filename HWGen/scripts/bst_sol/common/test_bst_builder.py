#!/usr/bin/env python3
"""
Test BST Builder functionality
Usage: python test_bst_builder.py --config config.vh --top top.v
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.bst_sol.common.config_parser import VerilogConfigParser
from scripts.bst_sol.common.bst_builder import BSTBuilder


def diagnose_ports(builder, ports):
    """Diagnose port validation issues"""
    print("\n  Diagnostic Info:")
    
    # Check clock and reset
    input_names = [p.name for p in ports['inputs']]
    print(f"    Clock: {'✓' if 'clk' in input_names else '✗'} clk")
    print(f"    Reset: {'✓' if 'rst_n' in input_names else '✗'} rst_n")
    
    # Check AXI input channels
    channels = builder.get_axi_channels(ports)
    print(f"    AXI Input Channels: {len(channels)}")
    for ch in channels:
        print(f"      - Channel {ch['index']}: {ch['data'].name}, {ch['valid'].name}, {ch['ready'].name}")
    
    if not channels:
        print("      ✗ No AXI input channels found!")
        print("      Input ports found:")
        for p in ports['inputs']:
            print(f"        - {p.name} ({p.direction}, {p.width}-bit)")
    
    # Check AXI output
    output_axi = builder.get_output_ports(ports)
    required = {'tdata', 'tvalid', 'tready'}
    print(f"    AXI Output: {list(output_axi.keys())}")
    for sig in required:
        status = '✓' if sig in output_axi else '✗'
        print(f"      {status} {sig}")
    
    if not all(sig in output_axi for sig in required):
        print("      ✗ Missing AXI output signals!")
        print("      Output ports found:")
        for p in ports['outputs']:
            print(f"        - {p.name} ({p.direction}, {p.width}-bit)")


def main():
    parser = argparse.ArgumentParser(description='Test BST Builder')
    parser.add_argument('--config', '-c', required=True, help='Path to config.vh')
    parser.add_argument('--top', '-t', required=True, help='Path to top.v')
    parser.add_argument('--show-instance', '-i', action='store_true', help='Show instance code')
    parser.add_argument('--show-signals', '-s', action='store_true', help='Show signal declarations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    config_file = Path(args.config)
    top_file = Path(args.top)
    
    if not config_file.exists():
        print(f"Error: {config_file} not found")
        return 1
    if not top_file.exists():
        print(f"Error: {top_file} not found")
        return 1
    
    print(f"\n{'='*70}")
    print("BST Builder Test")
    print(f"{'='*70}\n")
    
    # Parse config
    print("[1/4] Parsing config...")
    try:
        cfg_parser = VerilogConfigParser(str(config_file))
        cfg = cfg_parser.get_config()
        print(f"  Config: {cfg.n_parameters} params -> {cfg.n_solutions} solutions ({cfg.data_format})")
    except Exception as e:
        print(f"  Failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Create builder
    print("\n[2/4] Creating builder...")
    try:
        class TempConfig:
            def __init__(self, out_dir):
                self.output_dir = out_dir
        
        builder = BSTBuilder(TempConfig(top_file.parent))
        print(f"  Output: {builder.output_dir}")
    except Exception as e:
        print(f"  Failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Parse top module
    print("\n[3/4] Parsing top module...")
    try:
        ports = builder.parse_top_module(top_file)
        print(f"  Found {len(ports['inputs'])} inputs, {len(ports['outputs'])} outputs")
        
        if args.verbose:
            print("\n  All ports:")
            print("  Inputs:")
            for p in ports['inputs']:
                print(f"    - {p.name:30s} {p.direction:6s} {p.width:3d}-bit")
            print("  Outputs:")
            for p in ports['outputs']:
                print(f"    - {p.name:30s} {p.direction:6s} {p.width:3d}-bit")
    except Exception as e:
        print(f"  Failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Validate
    print("\n[4/4] Validating ports...")
    try:
        if builder.validate_ports(ports):
            print("  ✓ Validation passed")
        else:
            print("  ✗ Validation failed")
            diagnose_ports(builder, ports)
            if not args.show_instance and not args.show_signals:
                return 1
    except Exception as e:
        print(f"  Failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Summary
    print(f"\n{'='*70}")
    builder.print_port_summary(ports)
    
    # Optional outputs
    if args.show_instance:
        print(f"\n{'='*70}")
        print("Instance Code:")
        print(f"{'='*70}\n")
        module_name = top_file.stem
        print(builder.generate_instance(module_name, ports))
    
    if args.show_signals:
        print(f"\n{'='*70}")
        print("Signal Declarations:")
        print(f"{'='*70}\n")
        print(builder.generate_signal_declarations(ports))
    
    print(f"\n{'='*70}")
    print("✓ Test completed")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())