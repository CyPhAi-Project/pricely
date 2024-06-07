#!/usr/bin/env python
# coding: utf-8
"""
Learning Dyanmics and Lyapunov function for the Van der Pol oscillator
"""

from dreal import And, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 2
X_ROI = np.array([
    [-1.25, -1.25],  # Lower bounds
    [+1.25, +1.25]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 0.03125
NORM_LB = 0.2
NORM_UB = 1.2

MU = 1.0

def f_bbox(x: np.ndarray):
    # Actual dynamical system
    assert x.shape[1] == X_DIM
    x0, x1 = x[:, 0], x[:, 1]
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = -x1
    dxdt[:, 1] = x0 + MU*(x0**2-1)*x1
    return dxdt


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    Jacobian of RHS = [
        [           0,         -1],
        [ 1+2*μ*x0*x1, μ*(x0^2-1)]]
    We further upper bound the Jacobian by replacing 2*μ*x0*x1 with μ(x0^2+x1^2)
    and (x0^2-1)^2 by x0^4+1
    The Frobenious Norm is then derived as the square root of below:
        1 + (1 + μ(x0^2+x1^2))^2 + μ^2(x0^4 + 1)
    The supremum is at the vertex furthest away from (0, 0).
    The proof is skipped.
    """
    assert x_regions.ndim == 3
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    x0, x1 = abs_furtherest[:, 0], abs_furtherest[:, 1]
    sos = 1.0 + (1.0 + MU*(x0**2 + x1**2))**2 + MU**2*(x0**4 + 1)
    res = np.sqrt(sos)
    assert res.ndim == 1 and res.shape[0] == x_regions.shape[0]
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    x0, x1 = x_vars
    return [-x1,
            x0 + MU*(x0**2-1)*x1]
