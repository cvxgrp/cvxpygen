"""
rtl_writer_float.py - Floating-point extensions
"""
from typing import List, Optional
from pathlib import Path
from .rtl_writer import RTLWriter


class FloatingPointWriterMixin:
    """Floating-point specific RTL generation"""
    
    def write_fp_multiply(self, target: str, a: str, b: str,
                         data_width: int = 32,
                         latency: int = 9,
                         comment: Optional[str] = None):
        """
        Write floating-point multiplication using IP core
        
        Args:
            target: Output wire name
            a: First operand
            b: Second operand
            data_width: FP width (16, 32, 64)
            latency: Pipeline latency
            comment: Optional comment
        """
        instance_name = f"fp_mult_{target}"
        
        if comment:
            self.write_comment(comment)
        
        self.write_comment(f"FP multiply: {a} * {b}")
        self.write_line(f"wire [{data_width-1}:0] {target};")
        self.write_line(f"wire {target}_valid;")
        self.write_blank()
        
        self.write_line(f"fp_multiplier #(")
        self.indent()
        self.write_line(f".DATA_WIDTH({data_width}),")
        self.write_line(f".LATENCY({latency})")
        self.dedent()
        self.write_line(f") {instance_name} (")
        self.indent()
        self.write_line(".clk(clk),")
        self.write_line(".rst_n(rst_n),")
        self.write_line(f".a({a}),")
        self.write_line(f".b({b}),")
        self.write_line(f".result({target}),")
        self.write_line(f".valid({target}_valid)")
        self.dedent()
        self.write_line(");")
        self.write_blank()
    
    def write_fp_add(self, target: str, a: str, b: str,
                    data_width: int = 32,
                    latency: int = 12,
                    comment: Optional[str] = None):
        """
        Write floating-point addition using IP core
        
        Args:
            target: Output wire name
            a: First operand
            b: Second operand
            data_width: FP width
            latency: Pipeline latency
            comment: Optional comment
        """
        instance_name = f"fp_add_{target}"
        
        if comment:
            self.write_comment(comment)
        
        self.write_comment(f"FP add: {a} + {b}")
        self.write_line(f"wire [{data_width-1}:0] {target};")
        self.write_line(f"wire {target}_valid;")
        self.write_blank()
        
        self.write_line(f"fp_adder #(")
        self.indent()
        self.write_line(f".DATA_WIDTH({data_width}),")
        self.write_line(f".LATENCY({latency})")
        self.dedent()
        self.write_line(f") {instance_name} (")
        self.indent()
        self.write_line(".clk(clk),")
        self.write_line(".rst_n(rst_n),")
        self.write_line(f".a({a}),")
        self.write_line(f".b({b}),")
        self.write_line(f".result({target}),")
        self.write_line(f".valid({target}_valid)")
        self.dedent()
        self.write_line(");")
        self.write_blank()
    
    def write_fp_compare(self, decision_var: str, a: str, b: str,
                        data_width: int = 32,
                        latency: int = 1,
                        comment: Optional[str] = None):
        """
        Write floating-point comparison (a <= b)
        
        Args:
            decision_var: Output boolean wire
            a: First operand
            b: Second operand
            data_width: FP width
            latency: Comparison latency
            comment: Optional comment
        """
        instance_name = f"fp_cmp_{decision_var}"
        
        if comment:
            self.write_comment(comment)
        
        self.write_comment(f"FP compare: {a} <= {b}")
        self.write_line(f"wire {decision_var};")
        self.write_line(f"wire {decision_var}_valid;")
        self.write_blank()
        
        self.write_line(f"fp_comparator #(")
        self.indent()
        self.write_line(f".DATA_WIDTH({data_width}),")
        self.write_line(f".LATENCY({latency})")
        self.dedent()
        self.write_line(f") {instance_name} (")
        self.indent()
        self.write_line(".clk(clk),")
        self.write_line(".rst_n(rst_n),")
        self.write_line(f".a({a}),")
        self.write_line(f".b({b}),")
        self.write_line(f".result({decision_var}),")
        self.write_line(f".valid({decision_var}_valid)")
        self.dedent()
        self.write_line(");")
        self.write_blank()
    
    def write_fp_mac(self, target: str, coeffs: List[str], values: List[str],
                    data_width: int = 32,
                    mult_latency: int = 9,
                    add_latency: int = 12,
                    comment: Optional[str] = None):
        """
        Write multiply-accumulate for floating-point
        target = sum(coeffs[i] * values[i])
        
        Uses pipelined tree structure for efficiency
        
        Args:
            target: Output wire name
            coeffs: List of coefficient signals
            values: List of value signals
            data_width: FP width
            mult_latency: Multiplier latency
            add_latency: Adder latency
            comment: Optional comment
        """
        if len(coeffs) != len(values):
            raise ValueError("Coefficient and value lists must match")
        
        n_terms = len(coeffs)
        instance_name = f"fp_mac_{target}"
        
        if comment:
            self.write_comment(comment)
        
        self.write_comment(f"FP MAC: {n_terms} terms")
        
        # Declare coefficient and value arrays
        self.write_line(f"wire [{data_width-1}:0] mac_coeffs [{n_terms-1}:0];")
        self.write_line(f"wire [{data_width-1}:0] mac_values [{n_terms-1}:0];")
        
        # Assign array elements
        for i, (coeff, val) in enumerate(zip(coeffs, values)):
            self.write_line(f"assign mac_coeffs[{i}] = {coeff};")
            self.write_line(f"assign mac_values[{i}] = {val};")
        
        self.write_blank()
        
        # Instantiate MAC module
        self.write_line(f"wire [{data_width-1}:0] {target};")
        self.write_line(f"wire {target}_valid;")
        self.write_blank()
        
        self.write_line(f"fp_dot_product #(")
        self.indent()
        self.write_line(f".DATA_WIDTH({data_width}),")
        self.write_line(f".N_TERMS({n_terms}),")
        self.write_line(f".MULT_LATENCY({mult_latency}),")
        self.write_line(f".ADD_LATENCY({add_latency})")
        self.dedent()
        self.write_line(f") {instance_name} (")
        self.indent()
        self.write_line(".clk(clk),")
        self.write_line(".rst_n(rst_n),")
        self.write_line(".coeffs(mac_coeffs),")
        self.write_line(".values(mac_values),")
        self.write_line(f".result({target}),")
        self.write_line(f".valid({target}_valid)")
        self.dedent()
        self.write_line(");")
        self.write_blank()
    
    def write_fp_threshold_compare(self, decision_var: str,
                                   dot_product: str, threshold: str,
                                   data_width: int = 32,
                                   comment: Optional[str] = None):
        """
        Write threshold comparison for halfplane decision
        decision_var = (dot_product <= threshold)
        
        Args:
            decision_var: Output boolean wire
            dot_product: FP dot product result
            threshold: FP threshold value
            data_width: FP width
            comment: Optional comment
        """
        if comment:
            self.write_comment(comment)
        
        self.write_fp_compare(
            decision_var=decision_var,
            a=dot_product,
            b=threshold,
            data_width=data_width,
            comment="Halfplane decision"
        )
    
    def write_fp_pipeline_stage(self, target: str, source: str,
                               data_width: int = 32,
                               metadata_width: int = 32,
                               comment: Optional[str] = None):
        """
        Write pipeline register for FP data and metadata
        
        Args:
            target: Output register prefix
            source: Input signal prefix
            data_width: FP data width
            metadata_width: Metadata width
            comment: Optional comment
        """
        if comment:
            self.write_comment(comment)
        
        self.write_comment("Pipeline stage")
        self.write_line(f"reg [{data_width-1}:0] {target}_data;")
        self.write_line(f"reg [{metadata_width-1}:0] {target}_metadata;")
        self.write_line(f"reg {target}_valid;")
        self.write_blank()
        
        self.write_line("always @(posedge clk or negedge rst_n) begin")
        self.indent()
        self.write_line("if (!rst_n) begin")
        self.indent()
        self.write_line(f"{target}_data <= {data_width}'d0;")
        self.write_line(f"{target}_metadata <= {metadata_width}'d0;")
        self.write_line(f"{target}_valid <= 1'b0;")
        self.dedent()
        self.write_line("end else begin")
        self.indent()
        self.write_line(f"{target}_data <= {source}_data;")
        self.write_line(f"{target}_metadata <= {source}_metadata;")
        self.write_line(f"{target}_valid <= {source}_valid;")
        self.dedent()
        self.write_line("end")
        self.dedent()
        self.write_line("end")
        self.write_blank()


class FloatingPointRTLWriter(RTLWriter, FloatingPointWriterMixin):
    """RTL Writer with floating-point support"""
    
    def __init__(self, output_file: Path, data_width: int = 32,
                 mult_latency: int = 9, add_latency: int = 12):
        """
        Initialize floating-point RTL writer
        
        Args:
            output_file: Output Verilog file path
            data_width: FP width (16, 32, 64)
            mult_latency: Multiplier pipeline latency
            add_latency: Adder pipeline latency
        """
        super().__init__(output_file)
        self.data_width = data_width
        self.mult_latency = mult_latency
        self.add_latency = add_latency
        
        # Derive FP format parameters
        if data_width == 16:
            self.exp_width = 5
            self.mant_width = 10
        elif data_width == 32:
            self.exp_width = 8
            self.mant_width = 23
        elif data_width == 64:
            self.exp_width = 11
            self.mant_width = 52
        else:
            raise ValueError(f"Unsupported FP width: {data_width}")
    
    def write_multiply(self, target: str, a: str, b: str,
                      signed: bool = True, comment: Optional[str] = None):
        """Override to use floating-point multiply"""
        self.write_fp_multiply(
            target=target,
            a=a,
            b=b,
            data_width=self.data_width,
            latency=self.mult_latency,
            comment=comment
        )
    
    def write_add(self, target: str, a: str, b: str,
                 signed: bool = True, comment: Optional[str] = None):
        """Floating-point addition wrapper"""
        self.write_fp_add(
            target=target,
            a=a,
            b=b,
            data_width=self.data_width,
            latency=self.add_latency,
            comment=comment
        )
    
    def write_compare(self, decision_var: str, a: str, b: str,
                     comment: Optional[str] = None):
        """Floating-point comparison wrapper"""
        self.write_fp_compare(
            decision_var=decision_var,
            a=a,
            b=b,
            data_width=self.data_width,
            comment=comment
        )