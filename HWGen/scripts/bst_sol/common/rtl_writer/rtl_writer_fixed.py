"""
rtl_writer_fixed.py - Fixed-point extensions
"""
from typing import List, Optional
from pathlib import Path
from .rtl_writer import RTLWriter


class FixedPointWriterMixin:
    """Fixed-point specific RTL generation"""
    
    def write_fixed_multiply(self, target: str, a: str, b: str,
                            a_width: int, b_width: int,
                            frac_bits: int, 
                            result_width: int,
                            comment: Optional[str] = None):
        """
        Write fixed-point multiplication with proper bit extraction
        
        Q(m,n) * Q(p,q) -> Q(m+p, n+q), extract to target format
        
        Args:
            target: Output signal name
            a, b: Input signal names
            a_width, b_width: Input bit widths
            frac_bits: Fractional bits (same for both inputs)
            result_width: Output bit width
            comment: Optional comment
        """
        temp_width = a_width + b_width
        
        if comment:
            self.write_comment(comment)
        
        self.write_comment(f"Fixed multiply: {a} * {b} -> {target}")
        
        # Temporary product (full precision)
        self.write_line(f"wire signed [{temp_width-1}:0] mult_temp_{target};")
        self.write_line(f"assign mult_temp_{target} = $signed({a}) * $signed({b});")
        
        # Extract result (discard lower frac_bits to normalize)
        # Product has 2*FRAC_BITS, we want FRAC_BITS, so shift right by FRAC_BITS
        high_bit = frac_bits + result_width - 1
        low_bit = frac_bits
        
        self.write_line(f"assign {target} = mult_temp_{target}[{high_bit}:{low_bit}];")
        self.write_blank()
    
    def write_fixed_mac(self, target: str, coeffs: List[str], values: List[str],
                       data_width: int, frac_bits: int,
                       comment: Optional[str] = None):
        """
        Write multiply-accumulate for fixed-point
        target = sum(coeffs[i] * values[i])
        
        Note: This generates inline expression, suitable for always blocks
        """
        if len(coeffs) != len(values):
            raise ValueError(f"Coefficient count ({len(coeffs)}) != value count ({len(values)})")
        
        if comment:
            self.write_comment(comment)
        
        # Build MAC expression (products will be normalized by hardware)
        terms = []
        for coeff, val in zip(coeffs, values):
            terms.append(f"($signed({coeff}) * $signed({val}))")
        
        mac_expr = ' + '.join(terms)
        self.write_line(f"{target} = {mac_expr};")
    
    def write_fixed_threshold_compare(self, decision_var: str,
                                     dot_product: str, threshold: str,
                                     frac_bits: int,
                                     comment: Optional[str] = None):
        """
        Write threshold comparison for halfplane decision
        Scale threshold to match dot product precision
        
        Args:
            decision_var: Output boolean (1 if <= threshold)
            dot_product: Accumulated dot product (has FRAC_BITS precision)
            threshold: Threshold value (needs scaling)
            frac_bits: Fractional bits
        """
        if comment:
            self.write_comment(comment)
        
        self.write_comment("Halfplane decision (scale threshold to match dot product)")
        self.write_line(f"wire signed [31:0] thresh_scaled_{decision_var} = $signed({threshold}) <<< {frac_bits};")
        self.write_line(f"assign {decision_var} = {dot_product} <= thresh_scaled_{decision_var};")
        self.write_blank()


class FixedPointRTLWriter(RTLWriter, FixedPointWriterMixin):
    """RTL Writer with fixed-point support"""
    
    def __init__(self, output_file: Path, data_width: int = 16, 
                 frac_bits: int = 8):
        """
        Initialize fixed-point RTL writer
        
        Args:
            output_file: Output Verilog file path
            data_width: Total bit width (integer + fractional)
            frac_bits: Fractional bit width
        """
        super().__init__(output_file)
        self.data_width = data_width
        self.frac_bits = frac_bits
        self.int_bits = data_width - frac_bits
        
        # Validate configuration
        if frac_bits >= data_width:
            raise ValueError(f"Fractional bits ({frac_bits}) must be < data width ({data_width})")
        if frac_bits < 0 or data_width < 0:
            raise ValueError("Bit widths must be non-negative")
    
    def write_multiply(self, target: str, a: str, b: str,
                      signed: bool = True, comment: Optional[str] = None):
        """
        Override base multiply to use fixed-point version
        
        Assumes both inputs have same format: Q(int_bits, frac_bits)
        """
        if not signed:
            # Fallback to base class for unsigned
            super().write_multiply(target, a, b, signed, comment)
        else:
            self.write_fixed_multiply(
                target, a, b,
                self.data_width, self.data_width,
                self.frac_bits, self.data_width,
                comment
            )
    
    def get_format_string(self) -> str:
        """Return fixed-point format as string (for documentation)"""
        return f"Q{self.int_bits}.{self.frac_bits}"