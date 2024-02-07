"""
ODE from Equation (14) in "Formal Synthesis of Lyapunov Neural Networks"
DOI: 10.1109/LCSYS.2020.3005328
"""
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 3
X_ROI = np.array([
    [-3.75, -3.75, -3.75], # Lower bounds
    [+3.75, +3.75, +3.75]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 2**-10


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
    We further upper bound (-2+0.2xy)^2 with (2+0.2|x|y|)^2.
    The supremum is at (x, y) that is the furtherest from (0, 0),
    and therefore we just use the vertices of the rectangular region.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    assert abs_furtherest.ndim == 2 and abs_furtherest.shape[1] == X_DIM
    abs_x, abs_y, _ = abs_furtherest[:, 0], abs_furtherest[:, 1], abs_furtherest[:, 2]
    res = np.sqrt(1 + (0.1*abs_y**2)**2 + (2 + 0.2*abs_x*abs_y)**2 + 1 + 1.5**2 + 1).squeeze()
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y, z = x_vars
    return [-x,
            -2*y + 0.1*x*(y**2) + z,
            -z - 1.5*y]
