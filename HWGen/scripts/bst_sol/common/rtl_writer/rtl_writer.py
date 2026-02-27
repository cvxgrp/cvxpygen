"""
Unified RTL Writer
Provides consistent code generation utilities for all implementations
"""

from typing import List, Optional, Union
from pathlib import Path
from datetime import datetime


class RTLWriter:
    """RTL code generation helper with automatic indentation and formatting"""
    
    def __init__(self, output_file: Union[Path, str]):
        self.output_file = Path(output_file)
        self.lines: List[str] = []
        self.indent_level = 0
        self.line_count = 0  # Track line count
    
    def write_header(self, module_name: str, description: str, **metadata):
        """Write file header with metadata"""
        self.lines.append("`timescale 1ns/1ps")
        self.lines.append("")
        self.lines.append("/" + "=" * 69)
        self.lines.append(f"// {module_name}")
        self.lines.append("/" + "=" * 69)
        self.lines.append(f"// {description}")
        self.lines.append(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata:
            self.lines.append("//")
            for key, value in metadata.items():
                self.lines.append(f"// {key}: {value}")
        
        self.lines.append("/" + "=" * 69)
        self.lines.append("")
    
    def write_include(self, file_path: str):
        """Write include directive"""
        self.lines.append(f'`include "{file_path}"')
    
    def write_blank(self, count: int = 1):
        """Write blank lines"""
        self.lines.extend([''] * count)
    
    def write_comment(self, text: str, style: str = 'line'):
        """Write comment with proper indentation"""
        indent = '    ' * self.indent_level
        if style == 'line':
            self.lines.append(f"{indent}// {text}")
        elif style == 'block':
            self.lines.append(f"{indent}/*")
            for line in text.split('\n'):
                self.lines.append(f"{indent} * {line}")
            self.lines.append(f"{indent} */")
    
    def write_section(self, title: str):
        """Write section separator"""
        indent = '    ' * self.indent_level
        sep_len = max(60, 70 - len(indent))
        self.lines.append(f"{indent}//" + "=" * sep_len)
        self.lines.append(f"{indent}// {title}")
        self.lines.append(f"{indent}//" + "=" * sep_len)
    
    def write_separator(self, title: str):
        """Write 76-equals separator (baseline style)"""
        indent = '    ' * self.indent_level
        self.lines.append(f"{indent}//" + "=" * 76)
        self.lines.append(f"{indent}// {title}")
        self.lines.append(f"{indent}//" + "=" * 76)
    
    def write_line(self, code: str):
        """Write indented code line"""
        indent = '    ' * self.indent_level
        self.lines.append(f"{indent}{code}")
    
    def indent(self):
        """Increase indentation level"""
        self.indent_level += 1
    
    def dedent(self):
        """Decrease indentation level"""
        self.indent_level = max(0, self.indent_level - 1)
    
    def write_port(self, direction: str, name: str, width: Optional[str] = None, 
                   signed: bool = False, last: bool = False):
        """Write port declaration
        
        Args:
            direction: 'input' or 'output' or 'output reg' or 'input wire' etc.
            name: Port name
            width: Bit range like '15:0' or '[15:0]'
            signed: Whether signal is signed
            last: Whether this is the last port (no comma)
        """
        indent = '    ' * self.indent_level
        type_str = 'signed ' if signed else ''
        
        # Handle width format
        if width:
            if not width.startswith('['):
                width = f"[{width}]"
            width_str = f"{width} "
        else:
            width_str = ""
        
        comma = '' if last else ','
        self.lines.append(f"{indent}{direction} {type_str}{width_str}{name}{comma}")
    
    def write_reg(self, name: str, width: Optional[str] = None, 
                  signed: bool = False, array: Optional[str] = None,
                  init_value: Optional[str] = None):
        """Write register declaration
        
        Args:
            name: Register name
            width: Bit range like '15:0' or '[15:0]'
            signed: Whether register is signed
            array: Array range like '0:7' or '[0:7]'
            init_value: Initial value (optional)
        """
        indent = '    ' * self.indent_level
        type_str = 'signed ' if signed else ''
        
        # Handle width format
        if width:
            if not width.startswith('['):
                width = f"[{width}]"
            width_str = f"{width} "
        else:
            width_str = ""
        
        # Handle array format
        if array:
            if not array.startswith('['):
                array = f"[{array}]"
            array_str = f" {array}"
        else:
            array_str = ""
        
        init_str = f" = {init_value}" if init_value else ""
        self.lines.append(f"{indent}reg {type_str}{width_str}{name}{array_str}{init_str};")
    
    def write_wire(self, name: str, width: Optional[str] = None, 
                   signed: bool = False, array: Optional[str] = None):
        """Write wire declaration
        
        Args:
            name: Wire name
            width: Bit range like '15:0' or '[15:0]'
            signed: Whether wire is signed
            array: Array range like '0:7' or '[0:7]'
        """
        indent = '    ' * self.indent_level
        type_str = 'signed ' if signed else ''
        
        # Handle width format
        if width:
            if not width.startswith('['):
                width = f"[{width}]"
            width_str = f"{width} "
        else:
            width_str = ""
        
        # Handle array format
        if array:
            if not array.startswith('['):
                array = f"[{array}]"
            array_str = f" {array}"
        else:
            array_str = ""
        
        self.lines.append(f"{indent}wire {type_str}{width_str}{name}{array_str};")
    
    def write_memory(self, name: str, width: str, depth: str, 
                    signed: bool = False):
        """Write memory declaration (reg array)"""
        indent = '    ' * self.indent_level
        type_str = 'signed ' if signed else ''
        
        # Handle width format
        if not width.startswith('['):
            width = f"[{width}]"
        
        self.lines.append(f"{indent}reg {type_str}{width} {name} [0:{depth}-1];")
    
    def begin_module(self, name: str, parameters: Optional[List[str]] = None):
        """Begin module declaration
        
        Args:
            name: Module name
            parameters: Optional list of parameter declarations like 'N_PARAMS = 4'
        """
        if parameters:
            self.write_line(f"module {name} #(")
            self.indent()
            for i, param in enumerate(parameters):
                comma = '' if i == len(parameters) - 1 else ','
                self.write_line(f"parameter {param}{comma}")
            self.dedent()
            self.write_line(")(")
        else:
            self.write_line(f"module {name} (")
        self.indent()
    
    def end_ports(self):
        """End port list"""
        self.dedent()
        self.write_line(");")
        self.write_blank()
    
    def end_module(self):
        """End module"""
        self.write_line("endmodule")
    
    def begin_always(self, sensitivity: str):
        """Begin always block"""
        self.write_line(f"always @({sensitivity}) begin")
        self.indent()
    
    def end_always(self):
        """End always block"""
        self.dedent()
        self.write_line("end")
    
    def begin_if(self, condition: str):
        """Begin if statement"""
        self.write_line(f"if ({condition}) begin")
        self.indent()
    
    def begin_else_if(self, condition: str):
        """Begin else if statement"""
        self.dedent()
        self.write_line(f"end else if ({condition}) begin")
        self.indent()
    
    def begin_else(self):
        """Begin else statement"""
        self.dedent()
        self.write_line("end else begin")
        self.indent()
    
    def end_if(self):
        """End if statement"""
        self.dedent()
        self.write_line("end")
    
    def begin_case(self, expression: str):
        """Begin case statement"""
        self.write_line(f"case ({expression})")
        self.indent()
    
    def write_case_item(self, value: str, statement: str):
        """Write case item"""
        self.write_line(f"{value}: {statement}")
    
    def write_case_default(self, statement: str):
        """Write default case"""
        self.write_line(f"default: {statement}")
    
    def end_case(self):
        """End case statement"""
        self.dedent()
        self.write_line("endcase")
    
    def begin_for(self, init: str, condition: str, increment: str):
        """Begin for loop"""
        self.write_line(f"for ({init}; {condition}; {increment}) begin")
        self.indent()
    
    def end_for(self):
        """End for loop"""
        self.dedent()
        self.write_line("end")
    
    def begin_generate(self):
        """Begin generate block"""
        self.write_line("generate")
        self.indent()
    
    def end_generate(self):
        """End generate block"""
        self.dedent()
        self.write_line("endgenerate")
    
    def write_assign(self, target: str, expression: str, comment: Optional[str] = None):
        """Write continuous assignment"""
        comment_str = f"  // {comment}" if comment else ""
        self.write_line(f"assign {target} = {expression};{comment_str}")
    
    def save(self):
        """Write to file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.line_count = len(self.lines)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
            f.write('\n')  # Ensure file ends with newline
        
        # Print summary
        file_size = self.output_file.stat().st_size
        print(f"✓ Generated: {self.output_file} ({self.line_count} lines, {file_size} bytes)")
    
    def get_content(self) -> str:
        """Get generated content as string (for testing)"""
        return '\n'.join(self.lines)