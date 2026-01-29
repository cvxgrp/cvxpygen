"""
Test script: Generate RTL fixed-point modules from existing config
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.bst_sol.common.config_parser import VerilogConfigParser
from scripts.bst_sol.fixed_point.generate_fixed import FixedPointGenerator


def find_config_file(input_dir: Path) -> Path:
    """Find the config .vh file in input directory"""
    search_paths = [
        input_dir / "include" / "pdaqp_config.vh",
        input_dir / "pdaqp_config.vh",
        input_dir / "include" / "*_config.vh",
    ]
    
    for pattern in search_paths:
        if '*' in str(pattern):
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                return matches[0]
        elif pattern.exists():
            return pattern
    
    raise FileNotFoundError(f"No config file found in {input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate RTL from existing config')
    parser.add_argument('input_dir', type=str, help='Input directory with config')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir).resolve()
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = input_dir / "rtl_generated"
    
    print("=" * 70)
    print("RTL Fixed-Point Module Generator")
    print("=" * 70)
    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find config file
    print("\n" + "=" * 70)
    print("Finding Config File")
    print("=" * 70)
    
    try:
        config_file = find_config_file(input_dir)
        print(f"✓ Found config: {config_file}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Parse and display config
    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    
    config_parser = VerilogConfigParser(str(config_file))
    config = config_parser.get_config()
    
    print(f"\n  Project Name:    {config.project_name}")
    print(f"  Parameters:      {config.n_parameters}")
    print(f"  Solutions:       {config.n_solutions}")
    print(f"  Tree Nodes:      {config.n_tree_nodes}")
    print(f"  BST Depth:       {config.estimated_bst_depth}")
    print(f"  Data Format:     {config.data_format}")
    print(f"  Data Width:      {config.data_width}")
    
    # Show fixed-point format details
    if config.data_format.startswith('fixed'):
        if config.halfplane_int_bits is not None:
            print(f"  Halfplane:       Q{config.halfplane_int_bits}.{config.halfplane_frac_bits}")
        if config.feedback_int_bits is not None:
            print(f"  Feedback:        Q{config.feedback_int_bits}.{config.feedback_frac_bits}")
        if config.output_int_bits is not None:
            print(f"  Output:          Q{config.output_int_bits}.{config.output_frac_bits}")
    
    # Check format compatibility
    if not config.data_format.startswith('fixed'):
        print(f"\n✗ Error: This generator only supports fixed-point formats")
        print(f"  Current format: {config.data_format}")
        print(f"  Please use floating-point generator for FP16/FP32")
        return 1
    
    # Generate RTL
    print("\n" + "=" * 70)
    print("Generating RTL Modules")
    print("=" * 70)
    
    try:
        generator = FixedPointGenerator(
            config_file=str(config_file),
            output_dir=str(output_dir)
        )
        
        # Run generation
        generated_files = generator.generate_all()
        
        # Scan output directory for generated Verilog files
        rtl_dir = output_dir / "rtl"
        if rtl_dir.exists():
            verilog_files = sorted(rtl_dir.glob("*.v"))
        else:
            verilog_files = []
        
        # Display results
        if not verilog_files:
            print("\n⚠ Warning: No .v files found in output directory")
            print(f"  Looked in: {rtl_dir}")
        else:
            print(f"\n✓ Successfully generated {len(verilog_files)} Verilog files:\n")
            
            total_size = 0
            for f in verilog_files:
                size = f.stat().st_size
                total_size += size
                print(f"  - {f.name:40s} {size:8,d} bytes")
            
            print(f"\n  Total: {total_size:,d} bytes")
        
        print(f"\n✓ Output directory: {output_dir}")
        
        # Show next steps
        print("\n" + "-"*70)
        print("Next Steps:")
        print("-"*70)
        print("1. Review generated RTL files:")
        if verilog_files:
            for f in verilog_files:
                print(f"   - {f.relative_to(output_dir)}")
        print("\n2. Run syntax check:")
        print(f"   cd {output_dir}")
        print(f"   iverilog -t null rtl/*.v")
        print("\n3. Run simulation (if testbench available)")
        print("\n4. Synthesize with FPGA toolchain")
        print("-"*70)
        
        print("\n" + "="*70)
        print("✓ RTL Generation COMPLETED")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
