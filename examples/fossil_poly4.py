"""
poly_4 in "FOSSIL: A Software Tool for the Formal Synthesis of LyapunovFunctions and Barrier Certificates using Neural Networks"
DOI: 10.1145/3447928.3456646

Adapted from
https://github.com/oxford-oxcav/fossil/blob/main/experiments/benchmarks/models.py
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 2
X_NORM_LB = 0.1
X_NORM_UB = 1.0
X_LIM = np.array([
    [-X_NORM_UB]*X_DIM, # Lower bounds
    [+X_NORM_UB]*X_DIM  # Upper bounds
])
assert X_LIM.shape == (2, X_DIM)
ABS_X_LB = X_NORM_LB/np.sqrt(X_DIM)


def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x, y = q[:, 0], q[:, 1]
    dq = np.zeros_like(q)
    dq[:, 0] = -x - 1.5 * x**2 * y**3
    dq[:, 1] = -(y**3) + 0.5 * x**3 * y**2
    return dq


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [   -1-3*x*y^3, 4.5*x^2*y^2],
        [1.5*x^2*y^2, -3y^2+x^3*y]]
    We further upper bound (-3y^2+x^3*y)^2 with (|y||x|^3+3|y|^2)^2.
    The supremum is at (x, y) that is the furtherest from (0, 0),
    and therefore we just use the vertices of the rectangular region.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    assert abs_furtherest.ndim == 2 and abs_furtherest.shape[1] == X_DIM
    x, y = abs_furtherest[:, 0], abs_furtherest[:, 1]
    res = np.sqrt(
        (1 + 3 * x * y**3)**2 + (4.5 * x**2 * y**2)**2 +
        (1.5 * x**2 * y**2)**2 + (x**3 * y + 3 * y**2)**2)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y = x_vars
    return [-x - 1.5 * x**2 * y**3, -(y**3) + 0.5 * x**3 * y**2]
