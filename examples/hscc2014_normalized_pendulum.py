"""
ODE from Section 5.1 in "Simulation-guided Lyapunov Analysis for Hybrid Dynamical Systems"
DOI: 10.1145/2562059.2562139
"""

from dreal import sin as Sin, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 2
X_NORM_UB = 1.0
# X_NORM_LB = 0.01
X_LIM = np.array([
    [-X_NORM_UB, -X_NORM_UB],  # Lower bounds
    [+X_NORM_UB, +X_NORM_UB]  # Upper bounds
])
assert X_LIM.shape == (2, X_DIM)
ABS_X_LB = 2**-6


def f_bbox(x: np.ndarray):
    # Actual dynamical system
    assert x.shape[1] == X_DIM
    x0, x1 = x[:, 0], x[:, 1]
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = x1
    dxdt[:, 1] = -np.sin(x0) - x1
    return dxdt


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [        0,  1],
        [ -cos(x1), -1]]
    We further upper bound Frobenious Norm by replacing cos(x1) with 1.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    res = np.full(shape=x_regions.shape[0], fill_value=3.0)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x0, x1 = x_vars
    return [x1,
            -Sin(x0) - x1]
