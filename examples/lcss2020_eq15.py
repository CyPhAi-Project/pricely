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
LIP_CAP = 20
ABS_X_LB = 2**-10


def f_bbox(q: np.ndarray) -> np.ndarray:
    assert(q.shape[1] == X_DIM)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    dq = np.zeros_like(q)
    dq[:, 0] = -3*x - 0.1*x*(y**3)
    dq[:, 1] = -y + z
    dq[:, 2] = -z
    return dq


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Spectral Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [-3-0.1y^3, -0.3xy^2,  0],
        [        0,       -1,  1],
        [        0,        0, -1]]
    We further upper bound (-3-0.1y^3)^2 with (3+0.1|y|^3)^2.
    The supremum is at (x, y) that is the furtherest from (0, 0),
    and therefore we just use the vertices of the rectangular region.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    assert abs_furtherest.ndim == 2 and abs_furtherest.shape[1] == X_DIM
    abs_x, abs_y, _ = abs_furtherest[:, 0], abs_furtherest[:, 1], abs_furtherest[:, 2]
    res = np.sqrt((3 + 0.1*abs_y**3)**2+ (0.3*abs_x*abs_y**2)**2 + 1 + 1 + 1).squeeze()
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x, y, z = x_vars
    return [-3*x - 0.1*x*(y**3),
            -y + z,
            -z]
