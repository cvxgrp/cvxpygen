from dataclasses import dataclass, field

import scipy.sparse as sp
import numpy as np


@dataclass
class Configuration:
    code_dir: str
    solver_name: str
    unroll: bool
    prefix: str


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
    p: dict = field(default_factory=dict)
    p_csc: dict[str, sp.csc_matrix] = field(default_factory=dict)
    p_id_to_mapping: dict[str, sp.csr_matrix] = field(default_factory=dict)
    p_id_to_changes: dict[str, bool] = field(default_factory=dict)
    p_id_to_size: dict[str, int] = field(default_factory=dict)
    nonzero_d: bool = True
    is_maximization: bool = False
    user_p_name_to_canon_outdated: dict[str, list[str]] = field(default_factory=dict)
    quad_obj: bool = True


@dataclass
class ParameterInfo:
    col_to_name_usp: dict[int, str]
    flat_usp: np.ndarray
    id_to_col: dict[int, int]
    ids: list[int]
    name_to_shape: dict[str, tuple]
    name_to_size_usp: dict[str, int]
    name_to_sparsity: dict[str, np.ndarray]
    name_to_sparsity_type: dict[str, str]
    names: list[str]
    num: int
    sparsity_mask: np.ndarray
    writable: dict[str, np.ndarray]


@dataclass
class VariableInfo:
    name_to_offset: dict[str, int]
    name_to_indices: dict[str, np.ndarray]
    name_to_size: dict[str, int]
    sizes: list[int]
    name_to_shape: dict[str, tuple]
    name_to_init: dict[str, np.ndarray]


@dataclass
class PrimalVariableInfo(VariableInfo):
    name_to_sym: dict[str, bool]
    sym: list[bool]


@dataclass
class DualVariableInfo(VariableInfo):
    name_to_vec: dict[str, str]


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
    parameters_outdated: list[str]
    operator: str = None
    functions_if_false: list[str] = None
    extra_condition: str = None
    extra_condition_operator: str = None


@dataclass
class ParameterUpdateLogic:
    update_pending_logic: UpdatePendingLogic
    function_call: str
