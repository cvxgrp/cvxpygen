"""
BST Builder - Core build orchestration
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import logging


@dataclass
class PortInfo:
    """Verilog port information"""
    name: str
    direction: str  # 'input' or 'output'
    width: int      # bit width
    msb: int = 0    # MSB index
    lsb: int = 0    # LSB index
    
    @property
    def width_str(self) -> str:
        """Get width string for declaration"""
        if self.width == 1:
            return ""
        return f"[{self.msb}:{self.lsb}] "
    
    @property
    def range_str(self) -> str:
        """Get range string [msb:lsb]"""
        if self.width == 1:
            return ""
        return f"[{self.msb}:{self.lsb}]"


class BSTBuilder:
    """Main builder for BST LUT generation"""
    
    def __init__(self, config):
        """
        Args:
            config: Config object with output_dir attribute
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_top_module(self, top_file: Path) -> Dict[str, List[PortInfo]]:
        """Parse top module and extract port information"""
        if not top_file.exists():
            raise FileNotFoundError(f"Top module not found: {top_file}")
        
        with open(top_file, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = self._remove_comments(content)
        
        # Extract module ports
        module_pattern = r'module\s+\w+\s*\((.*?)\);'
        match = re.search(module_pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Cannot parse module in {top_file}")
        
        ports_section = match.group(1)
        
        # Parse each port
        inputs = []
        outputs = []
        
        port_pattern = r'(input|output)\s+wire\s+(?:\[(\d+):(\d+)\]\s+)?(\w+)'
        
        for match in re.finditer(port_pattern, ports_section):
            direction = match.group(1)
            msb = int(match.group(2)) if match.group(2) else 0
            lsb = int(match.group(3)) if match.group(3) else 0
            name = match.group(4)
            width = (msb - lsb + 1) if match.group(2) else 1
            
            port = PortInfo(name, direction, width, msb, lsb)
            
            if direction == 'input':
                inputs.append(port)
            else:
                outputs.append(port)
        
        return {'inputs': inputs, 'outputs': outputs}
    
    def _remove_comments(self, content: str) -> str:
        """Remove Verilog comments"""
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove single-line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        return content
    
    def detect_axi_style(self, ports: Dict[str, List[PortInfo]]) -> str:
        """
        Detect AXI interface style
        Returns: 'packed', 'multi_channel', or 'multi_parallel'
        """
        inputs = ports['inputs']
        
        # Check for packed style (single s_axis_tdata)
        has_packed = any(p.name == 's_axis_tdata' for p in inputs)
        
        # Check for multi-channel style (s_axis_tdata_N)
        has_multi_channel = any(
            re.match(r's_axis_tdata_\d+$', p.name) for p in inputs
        )
        
        # Check for multi-parallel style (s_axis_tdata_N_tdata)
        has_multi_parallel = any(
            re.match(r's_axis_tdata_\d+_tdata$', p.name) for p in inputs
        )
        
        if has_packed:
            return 'packed'
        elif has_multi_parallel:
            return 'multi_parallel'
        elif has_multi_channel:
            return 'multi_channel'
        else:
            return 'unknown'
    
    def get_axi_channels(self, ports: Dict[str, List[PortInfo]]) -> List[Dict]:
        """Group AXI input ports by channel index"""
        style = self.detect_axi_style(ports)
        
        if style == 'packed':
            return self._get_packed_axi_channel(ports)
        elif style == 'multi_parallel':
            return self._get_multi_parallel_channels(ports)
        elif style == 'multi_channel':
            return self._get_multi_axi_channels(ports)
        else:
            return []
    
    def _get_packed_axi_channel(self, ports: Dict[str, List[PortInfo]]) -> List[Dict]:
        """Get single packed AXI channel"""
        inputs = ports['inputs']
        outputs = ports['outputs']
        
        data_port = next((p for p in inputs if p.name == 's_axis_tdata'), None)
        valid_port = next((p for p in inputs if p.name == 's_axis_tvalid'), None)
        ready_port = next((p for p in outputs if p.name == 's_axis_tready'), None)
        
        if data_port and valid_port and ready_port:
            return [{
                'index': '0',
                'data': data_port,
                'valid': valid_port,
                'ready': ready_port,
                'style': 'packed'
            }]
        return []
    
    def _get_multi_parallel_channels(self, ports: Dict[str, List[PortInfo]]) -> List[Dict]:
        """Get multi-parallel AXI channels (s_axis_tdata_N_tdata format)"""
        inputs = ports['inputs']
        outputs = ports['outputs']
        channels = []
        
        # Find all data ports
        data_ports = [p for p in inputs if re.match(r's_axis_tdata_\d+_tdata$', p.name)]
        
        for data_port in data_ports:
            # Extract channel index (from s_axis_tdata_N_tdata)
            match = re.match(r's_axis_tdata_(\d+)_tdata$', data_port.name)
            if match:
                idx = match.group(1)
                
                # Find corresponding valid (s_axis_tdata_N_tvalid)
                valid_port = next(
                    (p for p in inputs if p.name == f's_axis_tdata_{idx}_tvalid'), 
                    None
                )
                
                # Ready signal is shared (s_axis_tready)
                ready_port = next(
                    (p for p in outputs if p.name == 's_axis_tready'), 
                    None
                )
                
                if valid_port and ready_port:
                    channels.append({
                        'index': idx,
                        'data': data_port,
                        'valid': valid_port,
                        'ready': ready_port,
                        'style': 'multi_parallel'
                    })
        
        return sorted(channels, key=lambda x: int(x['index']))
    
    def _get_multi_axi_channels(self, ports: Dict[str, List[PortInfo]]) -> List[Dict]:
        """Get multiple AXI channels (s_axis_tdata_N format)"""
        inputs = ports['inputs']
        outputs = ports['outputs']
        channels = []
        
        # Find all data ports
        data_ports = [p for p in inputs if re.match(r's_axis_tdata_\d+$', p.name)]
        
        for data_port in data_ports:
            # Extract channel index
            match = re.match(r's_axis_tdata_(\d+)$', data_port.name)
            if match:
                idx = match.group(1)
                
                # Find corresponding valid and ready
                valid_port = next((p for p in inputs if p.name == f's_axis_tvalid_{idx}'), None)
                ready_port = next((p for p in outputs if p.name == f's_axis_tready_{idx}'), None)
                
                if valid_port and ready_port:
                    channels.append({
                        'index': idx,
                        'data': data_port,
                        'valid': valid_port,
                        'ready': ready_port,
                        'style': 'multi_channel'
                    })
        
        return sorted(channels, key=lambda x: int(x['index']))
    
    def get_output_ports(self, ports: Dict[str, List[PortInfo]]) -> Dict[str, PortInfo]:
        """Extract AXI output ports (m_axis_*)"""
        outputs = ports['outputs']
        inputs = ports['inputs']
        
        result = {}
        
        # Check for multi-parallel output (m_axis_tdata_N_tdata)
        parallel_outputs = [p for p in outputs if re.match(r'm_axis_tdata_\d+_tdata$', p.name)]
        if parallel_outputs:
            # Multi-parallel output
            for port in outputs:
                if re.match(r'm_axis_tdata_\d+_tdata$', port.name):
                    if 'tdata' not in result:
                        result['tdata'] = []
                    result['tdata'].append(port)
                elif re.match(r'm_axis_tdata_\d+_tvalid$', port.name):
                    if 'tvalid' not in result:
                        result['tvalid'] = []
                    result['tvalid'].append(port)
            
            # Shared tready
            ready_port = next((p for p in inputs if p.name == 'm_axis_tready'), None)
            if ready_port:
                result['tready'] = ready_port
        else:
            # Single output or standard format
            for port in outputs:
                if port.name.startswith('m_axis_'):
                    signal_type = port.name.split('_')[-1]
                    result[signal_type] = port
            
            # tready is input from downstream
            for port in inputs:
                if port.name == 'm_axis_tready':
                    result['tready'] = port
        
        return result
    
    def generate_instance(self, module_name: str, ports: Dict[str, List[PortInfo]]) -> str:
        """Generate module instantiation code"""
        lines = []
        lines.append(f"{module_name} u_dut (")
        
        # Clock and reset
        lines.append("    .clk(clk),")
        lines.append("    .rst_n(rst_n),")
        lines.append("")
        
        # AXI input channels
        axi_channels = self.get_axi_channels(ports)
        style = self.detect_axi_style(ports)
        
        if style == 'packed':
            lines.append("    // AXI Input (packed)")
            ch = axi_channels[0]
            lines.append(f"    .{ch['data'].name}({ch['data'].name}),")
            lines.append(f"    .{ch['valid'].name}({ch['valid'].name}),")
            lines.append(f"    .{ch['ready'].name}({ch['ready'].name}),")
        elif style == 'multi_parallel':
            lines.append("    // AXI Input (multi-parallel)")
            for i, ch in enumerate(axi_channels):
                lines.append(f"    .{ch['data'].name}({ch['data'].name}),")
                lines.append(f"    .{ch['valid'].name}({ch['valid'].name}),")
            # Shared ready
            if axi_channels:
                lines.append(f"    .{axi_channels[0]['ready'].name}({axi_channels[0]['ready'].name}),")
            lines.append("")
        else:
            for i, ch in enumerate(axi_channels):
                lines.append(f"    // AXI Input {ch['index']}")
                lines.append(f"    .{ch['data'].name}({ch['data'].name}),")
                lines.append(f"    .{ch['valid'].name}({ch['valid'].name}),")
                lines.append(f"    .{ch['ready'].name}({ch['ready'].name}),")
                lines.append("")
        
        # AXI output
        output_axi = self.get_output_ports(ports)
        if output_axi:
            lines.append("    // AXI Output")
            
            # Handle multi-parallel output
            if isinstance(output_axi.get('tdata'), list):
                for port in output_axi['tdata']:
                    lines.append(f"    .{port.name}({port.name}),")
                for port in output_axi.get('tvalid', []):
                    lines.append(f"    .{port.name}({port.name}),")
                if 'tready' in output_axi:
                    port = output_axi['tready']
                    lines.append(f"    .{port.name}({port.name})")
            else:
                # Single output
                out_signals = ['tdata', 'tvalid', 'tready']
                for i, sig in enumerate(out_signals):
                    if sig in output_axi:
                        port = output_axi[sig]
                        comma = "" if i == len(out_signals) - 1 else ","
                        lines.append(f"    .{port.name}({port.name}){comma}")
        
        lines.append(");")
        return "\n".join(lines)
    
    def generate_signal_declarations(self, ports: Dict[str, List[PortInfo]], 
                                     exclude: List[str] = None) -> str:
        """Generate signal declarations for testbench"""
        if exclude is None:
            exclude = ['clk', 'rst_n']
        
        lines = []
        
        # Input signals (reg in testbench)
        input_lines = []
        for port in ports['inputs']:
            if port.name not in exclude:
                input_lines.append(f"reg {port.width_str}{port.name};")
        
        if input_lines:
            lines.append("// Input signals")
            lines.extend(input_lines)
            lines.append("")
        
        # Output signals (wire in testbench)
        output_lines = []
        for port in ports['outputs']:
            if port.name not in exclude:
                output_lines.append(f"wire {port.width_str}{port.name};")
        
        if output_lines:
            lines.append("// Output signals")
            lines.extend(output_lines)
            lines.append("")
        
        return "\n".join(lines)
    
    def validate_ports(self, ports: Dict[str, List[PortInfo]]) -> bool:
        """Validate port structure"""
        # Check clock and reset
        input_names = [p.name for p in ports['inputs']]
        if 'clk' not in input_names or 'rst_n' not in input_names:
            return False
        
        # Check AXI input channels
        channels = self.get_axi_channels(ports)
        if not channels:
            return False
        
        # Check AXI output
        output_axi = self.get_output_ports(ports)
        if not output_axi:
            return False
        
        # Validate required signals exist
        if isinstance(output_axi.get('tdata'), list):
            # Multi-parallel output
            return ('tdata' in output_axi and 
                   'tvalid' in output_axi and 
                   'tready' in output_axi)
        else:
            # Single output
            required = {'tdata', 'tvalid', 'tready'}
            return all(sig in output_axi for sig in required)
    
    def print_port_summary(self, ports: Dict[str, List[PortInfo]]):
        """Print port summary"""
        style = self.detect_axi_style(ports)
        
        print("Port Summary:")
        print(f"{'='*70}")
        print(f"AXI Style: {style}")
        
        print("\nInputs:")
        for port in ports['inputs']:
            width_str = f"[{port.msb}:{port.lsb}]" if port.width > 1 else "      "
            print(f"  {port.direction:6s} {width_str:12s} {port.name}")
        
        print("\nOutputs:")
        for port in ports['outputs']:
            width_str = f"[{port.msb}:{port.lsb}]" if port.width > 1 else "      "
            print(f"  {port.direction:6s} {width_str:12s} {port.name}")
        
        channels = self.get_axi_channels(ports)
        if channels:
            if style == 'packed':
                ch = channels[0]
                print(f"\nAXI Input: Packed single port")
                print(f"  Data width: {ch['data'].width}-bit")
            elif style == 'multi_parallel':
                print(f"\nAXI Input: Multi-parallel ({len(channels)} ports)")
                for ch in channels:
                    print(f"  Port {ch['index']}: {ch['data'].width}-bit")
            else:
                print(f"\nAXI Input: {len(channels)} channels")
                for ch in channels:
                    print(f"  Channel {ch['index']}: {ch['data'].width}-bit")
        
        output_axi = self.get_output_ports(ports)
        if output_axi:
            if isinstance(output_axi.get('tdata'), list):
                print(f"\nAXI Output: Multi-parallel ({len(output_axi['tdata'])} ports)")
                for port in output_axi['tdata']:
                    print(f"  {port.name}: {port.width}-bit")
            elif 'tdata' in output_axi:
                print(f"\nAXI Output: {output_axi['tdata'].width}-bit")


def create_builder(config_path: Path):
    """Create builder from config file"""
    from .config_parser import ConfigParser
    
    parser = ConfigParser()
    config = parser.parse(config_path)
    return BSTBuilder(config)