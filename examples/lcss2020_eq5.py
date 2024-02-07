"""
ODE from Equation (5) in "Formal Synthesis of Lyapunov Neural Networks"
DOI: 10.1109/LCSYS.2020.3005328
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 2
X_ROI = np.array([
    [-2.0, -2.0], # Lower bounds
    [+2.0, +2.0]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 2**-10


def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x, y = q[:, 0], q[:, 1]
    dq = np.zeros_like(q)
    dq[:, 0] = -x + x*y
    dq[:, 1] = -y
    return dq


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [-1+y, x],
        [  0, -1]]
    Hence, the supremum is at (x, y) that is the furtherest from (0, 1).
    Given that the regions are hyperrectangles,
    we can compute the Frobenious norm for the vertice of the regions using
    the abs distance to (0, 1) for each axis.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    recentered = x_regions - np.asfarray([0.0, 1.0])
    abs_furtherest = np.max(np.abs(recentered), axis=1, keepdims=True)
    res = np.sqrt(np.sum(abs_furtherest**2, axis=2) + (-1)**2).squeeze()
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y = x_vars
    return [-x + x*y,
            -y]
