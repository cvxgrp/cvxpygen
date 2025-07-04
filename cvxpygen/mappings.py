from dataclasses import dataclass, field
from typing import Dict, List

import scipy.sparse as sp
import numpy as np


@dataclass
class Configuration:
    code_dir: str
    solver_name: str
    unroll: bool
    prefix: str
    gradient: bool
    gradient_two_stage: bool
    explicit: bool


@dataclass
class AffineMap:
    mapping_rows: list = field(default_factory=list)
    mapping: list = field(default_factory=list)
    sign: int = 1
    indices: list = field(default_factory=list)
    indptr: list = field(default_factory=list)
    shape = ()


@dataclass
class ParameterCanon:
    """Represents first affine form"""
    p: dict = field(default_factory=dict)
    p_csc: Dict[str, sp.csc_matrix] = field(default_factory=dict)
    p_id_to_mapping: Dict[str, sp.csr_matrix] = field(default_factory=dict)  # Represents A slice to canonical parameter
    p_id_to_changes: Dict[str, bool] = field(default_factory=dict)
    p_id_to_size: Dict[str, int] = field(default_factory=dict)
    nonzero_d: bool = True
    is_maximization: bool = False
    user_p_name_to_canon_outdated: Dict[str, List[str]] = field(default_factory=dict)
    quad_obj: bool = True
    th_mask: np.ndarray = None
    n_param_reduced: int = 0
    n_dual_reduced: int = 0


@dataclass
class ParameterInfo:
    """
    All info about a user defined parameter and how to convert from
    the user-defined parameter to the canonicalized vector that is
    passed to A.
    """
    col_to_name_usp: Dict[int, str]  # usp: user-defined sparsity
    flat_usp: np.ndarray
    id_to_col: Dict[int, int]  # Maps parameter id to column of the start of the parameter
    ids: List[int]
    name_to_shape: Dict[str, tuple]
    name_to_size_usp: Dict[str, int]
    name_to_sparsity: Dict[str, np.ndarray]
    name_to_sparsity_type: Dict[str, str]
    names: List[str]
    num: int
    sparsity_mask: np.ndarray
    writable: Dict[str, np.ndarray]
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class VariableInfo:
    name_to_offset: Dict[str, int]
    name_to_indices: Dict[str, np.ndarray]
    name_to_size: Dict[str, int]
    sizes: List[int]
    name_to_shape: Dict[str, tuple]
    name_to_init: Dict[str, np.ndarray]


@dataclass
class PrimalVariableInfo(VariableInfo):
    """Info for primal variable retrival from a canonical solution"""
    name_to_sym: Dict[str, bool]
    sym: List[bool]
    reduced: bool = False


@dataclass
class DualVariableInfo(VariableInfo):
    """Info for dual variable retrival from a canonical solution"""
    name_to_vec: Dict[str, str]


@dataclass
class ConstraintInfo:
    n_data_constr: int
    n_data_constr_mat: int
    mapping_rows_eq: np.ndarray
    mapping_rows_ineq: np.ndarray


@dataclass
class WorkspacePointerInfo:
    objective_value: str
    iterations: str
    status: str
    primal_residual: str
    dual_residual: str
    primal_solution: str
    dual_solution: str
    settings: str = None


@dataclass
class UpdatePendingLogic:
    parameters_outdated: List[str]
    operator: str = None
    functions_if_false: List[str] = None
    extra_condition: str = None
    extra_condition_operator: str = None


@dataclass
class ParameterUpdateLogic:
    update_pending_logic: UpdatePendingLogic
    function_call: str


@dataclass
class Canon:
    """
    All info for the ASA representation
    """
    prim_variable_info: PrimalVariableInfo
    dual_variable_info: DualVariableInfo
    parameter_info: ParameterInfo
    parameter_canon: ParameterCanon


@dataclass
class Setting:
    type: str
    default: str
    enabled: bool = True
    name_cvxpy: str = None
