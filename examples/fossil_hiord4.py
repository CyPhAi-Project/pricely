"""
hi-ord_4 in "FOSSIL: A Software Tool for the Formal Synthesis of LyapunovFunctions and Barrier Certificates using Neural Networks"
DOI: 10.1145/3447928.3456646

Adapted from
https://github.com/oxford-oxcav/fossil/blob/main/experiments/benchmarks/models.py
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 4
NORM_LB = 0.1
NORM_UB = 1.0
X_ROI = np.array([
    [-NORM_UB]*X_DIM, # Lower bounds
    [+NORM_UB]*X_DIM  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = NORM_LB/np.sqrt(X_DIM)

KNOWN_QUAD_LYA = np.array([
    [3.40203882e+00, 4.99460619e+00, 3.52756617e+00, 8.68055556e-04],
    [4.99460619e+00, 1.17344311e+01, 9.41395828e+00, 2.28941924e-03],
    [3.52756617e+00, 9.41395828e+00, 9.96493336e+00, 2.37176035e-03],
    [8.68055556e-04, 2.28941924e-03, 2.37176035e-03, 1.26224060e-04]])

def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x0, x1, x2, x3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dq = np.zeros_like(q)
    dq[:, 0] = x1
    dq[:, 1] = x2
    dq[:, 2] = x3
    dq[:, 3] = -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0
    return dq


LIP = np.linalg.norm(np.array([
    [   0,     1,     0,     0],
    [   0,     0,     1,     0],
    [   0,     0,     0,     1],
    [-576, -2400, -4180, -3980]]), ord=2)

def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Spectral Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [   0,    1,     0,     0],
        [   0,    0,     1,     0],
        [   0,    0,     0,     1],
        [-576,-2400, -4180, -3980]]
    The supremum of Spectal Norm is a constant.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    res = np.full(shape=x_regions.shape[0], fill_value=LIP)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x0, x1, x2, x3 = x_vars
    return [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]

