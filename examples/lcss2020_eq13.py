"""
ODE from Equation (13) in "Formal Synthesis of Lyapunov Neural Networks"
DOI: 10.1109/LCSYS.2020.3005328
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 2
NORM_LB = 0.1
NORM_UB = 1.0
X_ROI = np.array([
    [-NORM_UB]*X_DIM, # Lower bounds
    [+NORM_UB]*X_DIM  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = NORM_LB/np.sqrt(X_DIM)


def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x, y = q[:, 0], q[:, 1]
    dq = np.zeros_like(q)
    dq[:, 0] = -x + 2*(x**2)*y
    dq[:, 1] = -y
    return dq


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [-1+4xy, 2x^2],
        [     0,   -1]]
    We further upper bound (-1+4xy)^2 with (1+4|x||y|)^2
    The supremum is at (x, y) that is the furtherest from (0, 0),
    and therefore we just use the vertices of the rectangular region.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    assert abs_furtherest.ndim == 2 and abs_furtherest.shape[1] == X_DIM
    abs_x, abs_y = abs_furtherest[:, 0], abs_furtherest[:, 1]
    res = np.sqrt((1 + 4*abs_x*abs_y)**2 + 4*abs_x**4 + (-1)**2)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y = x_vars
    return [-x + 2*(x**2)*y,
            -y]
