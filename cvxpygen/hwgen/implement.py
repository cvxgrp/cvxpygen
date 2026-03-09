import numpy as np
from numpy.typing import NDArray

from cvxpygen.hwgen.hw_models import GEMV, DotProductLeq, PDAQPHWConfig, TreeWalkerFSM
from cvxpygen.hwgen.models import (
    DataFormat,
    MultiplyAddFix16,
    MultiplyAddFP32,
    PDAQPAlgoConfig,
)


def specialize(c: PDAQPAlgoConfig) -> PDAQPHWConfig:
    """Specialize the abstract algorithm to high-level hardware architecture."""

    assert c.n_feedbacks >= c.n_half_planes

    return PDAQPHWConfig(
        c.problem_size,
        TreeWalkerFSM(c.tree_nodes),
        DotProductLeq(c.half_planes),
        GEMV(c.feedbacks),
    )


def quantize(
    c: PDAQPHWConfig, data_format: DataFormat = DataFormat.FIXQ2_14
) -> PDAQPHWConfig:
    """Truncate the precision of the parameters to fixed point decimals."""
    assert isinstance(c.feedback_module.param, MultiplyAddFP32)
    assert isinstance(c.half_planes_module.param, MultiplyAddFP32)

    if data_format == DataFormat.FP32:
        # Do nothing
        return c

    def quantizeToFixQ2_14(arr: NDArray[np.float32]) -> NDArray[np.int16]:
        scale_factor = 2.0**14
        return np.clip(arr * scale_factor, -32678, 32767).astype(np.int16)

    halfplanes = c.half_planes_module.param
    c.half_planes_module.param = MultiplyAddFix16(
        quantizeToFixQ2_14(halfplanes.scale),
        quantizeToFixQ2_14(halfplanes.offset),
        Q=14,
    )

    feedbacks = c.feedback_module.param
    c.feedback_module.param = MultiplyAddFix16(
        quantizeToFixQ2_14(feedbacks.scale),
        quantizeToFixQ2_14(feedbacks.offset),
        Q=14,
    )

    return c
