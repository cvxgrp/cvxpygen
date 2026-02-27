#!/usr/bin/env python3
"""
PDAQP Interface Generator with 2-Tier Adaptive Packing and Cache Support
Generates top-level RTL with optimized AXI4-Stream interface
Tier 1: Spatial Multiplexing (default)
Tier 2: Time Division Multiplexing (IOB-constrained)
Supports both cache-enabled and non-cache BST designs
"""

import argparse
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Import from local interface package
from scripts.interface.config_parser import VerilogConfigParser
from scripts.interface.packing_strategy import PackingStrategy
from scripts.interface.top_generator import TopModuleGenerator
from scripts.interface.timing_generator import TimingHeaderGenerator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate PDAQP RTL top module with adaptive packing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default generation (Tier 1: spatial multiplexing, with input buffer)
  python generate_interface.py -c config/pdaqp_config.vh -o output/

  # Disable input buffering (lower latency, may affect timing)
  python generate_interface.py -c config/pdaqp_config.vh -o output/ --no-input-buffer

  # With IOB constraint (auto-selects Tier 2 TDM if needed)
  python generate_interface.py -c config/pdaqp_config.vh -o output/ --max-iob 200

  # Force cache mode (or disable cache)
  python generate_interface.py -c config/pdaqp_config.vh -o output/ --enable-cache
  python generate_interface.py -c config/pdaqp_config.vh -o output/ --disable-cache

  # Verbose output with strategy details
  python generate_interface.py -c config/pdaqp_config.vh -o output/ -v

Packing Tiers:
  Tier 1: Spatial Multiplexing (32/64/128/256/512-bit ports) - Best performance
  Tier 2: Time Division Multiplexing (configurable batching) - IOB-optimized

Input Buffering:
  Default: ENABLED  - Adds 1-cycle pipeline stage for better timing closure
  --no-input-buffer - Direct connection, lower latency but may affect timing

Cache Support:
  Default: AUTO-DETECT - Automatically detects cache from BST module
  --enable-cache       - Force cache mode (adds ROM and statistics ports)
  --disable-cache      - Force non-cache mode
        """
    )
    
    parser.add_argument('-c', '--config', required=True,
                       help='Path to pdaqp_config.vh')
    parser.add_argument('-o', '--output-dir', required=True,
                       help='Output directory for RTL files')
    parser.add_argument('--max-iob', type=int, default=None,
                       help='Maximum IOB pin count (default: unlimited, uses Tier 1)')
    parser.add_argument('--no-input-buffer', action='store_true',
                       help='Disable input buffering (lower latency, may affect timing)')
    parser.add_argument('--enable-cache', action='store_true',
                       help='Enable cache support in BST module (auto-detect if not specified)')
    parser.add_argument('--disable-cache', action='store_true',
                       help='Disable cache support (force non-cache mode)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with strategy details')
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments"""
    if not os.path.isfile(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if args.max_iob is not None and args.max_iob < 1:
        print(f"Error: max-iob must be positive, got {args.max_iob}")
        sys.exit(1)
    
    if args.enable_cache and args.disable_cache:
        print("Error: Cannot specify both --enable-cache and --disable-cache")
        sys.exit(1)


def create_output_directory(output_dir):
    """Create output directory structure"""
    rtl_dir = os.path.join(output_dir, 'rtl')
    include_dir = os.path.join(output_dir, 'include')
    os.makedirs(rtl_dir, exist_ok=True)
    os.makedirs(include_dir, exist_ok=True)
    return rtl_dir, include_dir


def print_summary(config, strategy, args, use_cache_mode):
    """Print generation summary"""
    print(f"\n{'='*70}")
    print("Configuration Summary")
    print(f"{'='*70}")
    print(f"  Parameters: {config.n_parameters}")
    print(f"  Solutions:  {config.n_solutions}")
    print(f"  Data format: {config.data_format}")
    print(f"  Input width:  {config.input_width} bits per parameter")
    print(f"  Output width: {config.output_width} bits per solution")
    print(f"  Total input:  {config.n_parameters * config.input_width} bits")
    print(f"  Total output: {config.n_solutions * config.output_width} bits")
    
    print(f"\n{'='*70}")
    print("Packing Strategy")
    print(f"{'='*70}")
    
    if args.verbose:
        print(strategy.get_summary())
    else:
        tier_names = {
            1: 'Spatial Multiplexing (Full Parallel)',
            2: 'Time Division Multiplexing (IOB Optimized)'
        }
        print(f"  Tier: {strategy.tier_used} - {tier_names[strategy.tier_used]}")
        
        input_info = strategy.input_strategy
        if input_info['mode'] == 'tdm':
            print(f"  Input:  TDM (batch={input_info['batch_size']}, "
                  f"batches={input_info['num_batches']}, {input_info['port_width']}-bit)")
        else:
            print(f"  Input:  {input_info['mode']} "
                  f"({input_info['num_ports']} port(s) × {input_info['port_width']}-bit)")
        
        output_info = strategy.output_strategy
        if output_info['mode'] == 'tdm':
            print(f"  Output: TDM (batch={output_info['batch_size']}, "
                  f"batches={output_info['num_batches']}, {output_info['port_width']}-bit)")
        else:
            print(f"  Output: {output_info['mode']} "
                  f"({output_info['num_ports']} port(s) × {output_info['port_width']}-bit)")
        
        if args.max_iob:
            estimated_iob = strategy._estimate_current_iob()
            utilization = (estimated_iob / args.max_iob) * 100
            print(f"  IOB Usage: {estimated_iob} / {args.max_iob} ({utilization:.1f}%)")
    
    # Input buffering status
    buffer_status = "DISABLED" if args.no_input_buffer else "ENABLED"
    print(f"\n{'='*70}")
    print("Input Buffering")
    print(f"{'='*70}")
    print(f"  Status: {buffer_status}")
    if args.no_input_buffer:
        print("  → Direct connection (lower latency, may affect timing)")
    else:
        print("  → Pipeline stage (+1 cycle latency, better timing closure)")
    
    # Cache status
    print(f"\n{'='*70}")
    print("BST Cache Support")
    print(f"{'='*70}")
    if use_cache_mode is None:
        print("  Mode: AUTO-DETECT (will check BST module)")
    elif use_cache_mode:
        print("  Mode: ENABLED (forced)")
        print("  → Includes ROM interfaces and statistics ports")
    else:
        print("  Mode: DISABLED (forced)")
        print("  → Non-cache BST design")
    
    print()


def main():
    """Main entry point"""
    args = parse_arguments()
    validate_inputs(args)
    
    print("=" * 70)
    print("PDAQP Interface Generator with 2-Tier Adaptive Packing")
    print("=" * 70)
    
    # Step 1: Parse configuration
    print("\n[1/4] Parsing configuration...")
    try:
        config = VerilogConfigParser(args.config)
        print(f"  ✓ Loaded: {config}")
    except Exception as e:
        print(f"  ✗ Error parsing configuration: {e}")
        sys.exit(1)
    
    # Step 2: Decide packing strategy
    print("\n[2/4] Deciding packing strategy...")
    try:
        strategy = PackingStrategy(
            n_params=config.n_parameters,
            n_sols=config.n_solutions,
            data_width=config.input_width,
            max_iob=args.max_iob
        )
        
        # Determine cache mode
        if args.enable_cache:
            use_cache_mode = True
        elif args.disable_cache:
            use_cache_mode = False
        else:
            use_cache_mode = None  # Auto-detect
        
        print_summary(config, strategy, args, use_cache_mode)
    except Exception as e:
        print(f"  ✗ Error creating packing strategy: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Generate timing header placeholder
    print("[3/4] Generating timing header...")
    try:
        rtl_dir, include_dir = create_output_directory(args.output_dir)
        
        timing_file = os.path.join(include_dir, 'pdaqp_timing.vh')
        timing_gen = TimingHeaderGenerator(config)
        estimated_latency = timing_gen.generate_placeholder(timing_file)
        
        print(f"  ✓ Created placeholder: {timing_file}")
        print(f"  ✓ Estimated latency: {estimated_latency} cycles")
    except Exception as e:
        print(f"  ✗ Error generating timing header: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Generate top module
    print("\n[4/4] Generating top module...")
    try:
        top_file = os.path.join(rtl_dir, 'pdaqp_top.v')
        
        # Calculate relative paths for includes
        config_abs = os.path.abspath(args.config)
        timing_abs = os.path.abspath(timing_file)
        rtl_abs = os.path.abspath(rtl_dir)
        
        config_rel_path = os.path.relpath(config_abs, rtl_abs)
        timing_rel_path = os.path.relpath(timing_abs, rtl_abs)
        
        # Generate top module with buffer and cache options
        enable_buffer = not args.no_input_buffer
        
        top_gen = TopModuleGenerator(
            config=config,
            strategy=strategy,
            enable_input_buffer=enable_buffer,
            use_cache=use_cache_mode  # Pass cache mode (None = auto-detect)
        )
        
        top_gen.generate(top_file, config_rel_path, timing_rel_path)
        
        print(f"  ✓ Created: {top_file}")
        
        # Report actual cache detection result
        if use_cache_mode is None:
            actual_cache = top_gen.use_cache
            cache_status = "CACHE-ENABLED" if actual_cache else "NON-CACHE"
            print(f"  ✓ Detected: {cache_status} BST design")
        
        buffer_msg = "with input buffering" if enable_buffer else "without input buffering (direct mode)"
        print(f"  ✓ Mode: {buffer_msg}")
        
    except Exception as e:
        print(f"  ✗ Error generating top module: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Success summary
    print("\n" + "=" * 70)
    print("✓ Generation completed successfully!")
    print("=" * 70)
    
    print("\nGenerated files:")
    print(f"  • {timing_file}  (placeholder)")
    print(f"  • {top_file}")
    
    print("\n⚠️  IMPORTANT:")
    print(f"  The file {os.path.basename(timing_file)} contains a placeholder value.")
    print("  It will be automatically updated when BST LUT is generated.")
    
    print("\nNext steps:")
    print("  1. Review generated RTL files")
    print("  2. Run BST LUT generator (will update pdaqp_timing.vh)")
    print("  3. Create testbench and run simulation")
    print("  4. Synthesize and verify timing")
    
    if strategy.tier_used == 2:
        print("\n💡 Info: Using Tier 2 (TDM)")
        print("  - Significantly reduced IOB usage")
        print("  - Multiple cycles needed for input/output transfer")
        print("  - Consider testbench support for batched transactions")
    
    if args.no_input_buffer:
        print("\n⚠️  Warning: Input buffering disabled")
        print("  Monitor timing paths carefully during synthesis.")
        print("  Consider adding timing constraints for combinational input paths.")
    
    if use_cache_mode is None and hasattr(top_gen, 'use_cache'):
        if top_gen.use_cache:
            print("\n💡 Info: Cache-enabled BST detected")
            print("  - ROM interfaces included for external data storage")
            print("  - Statistics ports available for performance monitoring")
            print("  - Consider connecting external ROM if needed")
    
    print()


if __name__ == '__main__':
    main()