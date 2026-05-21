# scripts/bst_sol/fixed_point_cached/bst_rom_interface.py
"""
BST ROM Interface Generator
Handles BST node data loading from ROM
"""

from ..common.config_parser import PDQAPConfig
from ..common.rtl_writer.rtl_writer import RTLWriter


class BSTROMInterfaceGenerator:
    """Generate BST ROM interface"""
    
    def __init__(self, config: PDQAPConfig):
        self.config = config
        self.n_params = self._get_param_dimension()
        self.param_width = config.data_width
        self.n_nodes = config.n_bst_nodes if hasattr(config, 'n_bst_nodes') else 1024
    
    def _get_param_dimension(self) -> int:
        if hasattr(self.config, 'n_parameters'):
            return self.config.n_parameters
        return 2
    
    def generate_rom_declarations(self, writer: RTLWriter):
        """Generate BST ROM declarations"""
        writer.write_section("BST Node ROM")
        writer.write_blank()
        
        node_idx_bits = max(10, (self.n_nodes - 1).bit_length())
        split_dim_bits = max(1, (self.n_params - 1).bit_length())
        project = self.config.project_name
        
        writer.write_comment("BST Node Structure: [split_dim, split_value, left, right, region, is_leaf]")
        writer.write_blank()
        
        # Split dimension ROM
        writer.write_reg(f"bst_split_dim_rom [{split_dim_bits-1}:0]", 
                        array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_split_dim.mem", bst_split_dim_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Split value ROM
        writer.write_reg(f"bst_split_value_rom [{self.param_width-1}:0]", 
                        array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_split_value.mem", bst_split_value_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Child pointers ROM
        writer.write_reg(f"bst_left_child_rom [{node_idx_bits-1}:0]", 
                        array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_left_child.mem", bst_left_child_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        writer.write_reg(f"bst_right_child_rom [{node_idx_bits-1}:0]", 
                        array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_right_child.mem", bst_right_child_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Region index ROM
        writer.write_reg(f"bst_region_idx_rom [{node_idx_bits-1}:0]", 
                        array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_region_idx.mem", bst_region_idx_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
        
        # Leaf flag ROM
        writer.write_reg("bst_is_leaf_rom", array=f"0:{self.n_nodes-1}")
        writer.write_line("initial begin")
        writer.indent()
        writer.write_line(f'$readmemh("../include/{project}_bst_is_leaf.mem", bst_is_leaf_rom);')
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()
    
    def generate_rom_read_logic(self, writer: RTLWriter):
        """Generate ROM read control"""
        writer.write_comment("BST ROM Read Logic")
        writer.write_line("always @(posedge clk or negedge rst_n) begin")
        writer.indent()
        
        writer.begin_if("!rst_n")
        writer.write_line("node_split_dim <= 0;")
        writer.write_line("node_split_value <= 0;")
        writer.write_line("node_left_child <= 0;")
        writer.write_line("node_right_child <= 0;")
        writer.write_line("node_region_idx <= 0;")
        writer.write_line("is_leaf <= 0;")
        
        writer.begin_else()
        writer.begin_if("bst_state == BST_LOAD_NODE")
        writer.write_line("node_split_dim <= bst_split_dim_rom[current_node];")
        writer.write_line("node_split_value <= bst_split_value_rom[current_node];")
        writer.write_line("node_left_child <= bst_left_child_rom[current_node];")
        writer.write_line("node_right_child <= bst_right_child_rom[current_node];")
        writer.write_line("node_region_idx <= bst_region_idx_rom[current_node];")
        writer.write_line("is_leaf <= bst_is_leaf_rom[current_node];")
        writer.end_if()
        writer.end_if()
        
        writer.dedent()
        writer.write_line("end")
        writer.write_blank()