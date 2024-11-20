"""
nonpoly_2 in "FOSSIL: A Software Tool for the Formal Synthesis of LyapunovFunctions and Barrier Certificates using Neural Networks"
DOI: 10.1145/3447928.3456646

Adapted from
https://github.com/oxford-oxcav/fossil/blob/main/experiments/benchmarks/models.py
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 3
X_NORM_LB = 0.1
X_NORM_UB = 1.0

X_LIM = np.array([
    [-X_NORM_UB]*X_DIM, # Lower bounds
    [+X_NORM_UB]*X_DIM  # Upper bounds
])
assert X_LIM.shape == (2, X_DIM)
ABS_X_LB = X_NORM_LB/np.sqrt(X_DIM)

KNOWN_QUAD_LYA = np.array([
    [0.25, 0, 0],
    [   0, 1, 0],
    [   0, 0, 1]])


def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    dq = np.zeros_like(q)
    dq[:, 0] = -x
    dq[:, 1] = -2*y + 0.1*x*(y**2) + z
    dq[:, 2] = -z - 1.5*y
    return dq


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [    -1,        0,  0],
        [0.1y^2, -2+0.2xy,  1],
        [     0,     -1.5, -1]]
    We further upper bound (-2+0.2xy)^2 with (2+0.2|x||y|)^2.
    The supremum is at (x, y) that is the furtherest from (0, 0),
    and therefore we just use the vertices of the rectangular region.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    assert abs_furtherest.ndim == 2 and abs_furtherest.shape[1] == X_DIM
    abs_x, abs_y, _ = abs_furtherest[:, 0], abs_furtherest[:, 1], abs_furtherest[:, 2]
    res = np.sqrt(1 + (0.1*abs_y**2)**2 + (2 + 0.2*abs_x*abs_y)**2 + 1 + 1.5**2 + 1)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y, z = x_vars
    return [-x,
            -2*y + 0.1*x*(y**2) + z,
            -z - 1.5*y]
