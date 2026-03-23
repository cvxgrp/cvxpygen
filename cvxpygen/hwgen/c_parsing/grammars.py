from parsimonious.grammar import Grammar

pdaqp_header_grammar = Grammar(r"""
file = header typedef* n_parameter n_solution function_def footer

header = pound pound
footer = pound
typedef = "typedef " c_type " " variable ";\n"
n_parameter = c_define "PDAQP_N_PARAMETER " digits newline
n_solution = c_define "PDAQP_N_SOLUTION " digits newline+
function_def = "void " ~r"[^;]+;" newline

c_type = "float" / "unsigned short"
variable = ~r"[a-z_]+"
digits = ~r"[0-9]+"

pound = ~r"#[^\n]+" newline+

c_define = "#define "
newline = "\n"
""")

pdaqp_c_grammar = Grammar(r"""
file = header float_array+ int_array+ algorithm

header = pound
float_array = "c_float_store " name size " = {\n" float_item+ "};\n"
int_array = "c_int " name size " = {\n" int_item+ "};\n"
algorithm = "void " name function_args algorithm_impl

float_item = casting decimal separator
int_item = casting digits separator
function_args = bracket_enclosed
algorithm_impl = "{\n" indented_code* "}\n"

name = ~r"[a-z_]+"
size = "[" digits "]"
casting = bracket_enclosed
indented_code = "    " ~"[^\n]*\n"

bracket_enclosed = ~r"\([^)]+\)"
decimal = ~r"-?[0-9]+\.[0-9]+(e-?[0-9]+)?"
digits = ~r"[0-9]+"
pound = ~r"#[^\n]+" newline+
newline = "\n"
separator = ",\n"
""")
