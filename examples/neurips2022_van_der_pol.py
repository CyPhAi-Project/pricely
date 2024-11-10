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


def nnet_lya(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.tanh(
        (0.51641756296157837
        + 0.75732171535491943 * np.tanh((-1.6187947988510132 + 2.0125248432159424 * x1 - 0.86828583478927612 * x2))
        - 1.6154271364212036 * np.tanh((-1.0764049291610718 + 0.26035198569297791 * x1 - 0.058430317789316177 * x2)) 
        + 1.2375599145889282 * np.tanh((-0.96464759111404419 - 0.50644028186798096 * x1 + 1.4162489175796509 * x2)) 
        + 0.41873458027839661 * np.tanh((-0.82901746034622192 + 2.5682404041290283 * x1 - 1.2206004858016968 * x2)) 
        - 0.89795422554016113 * np.tanh((0.98988056182861328 + 0.83175277709960938 * x1 + 1.0546237230300903 * x2)) 
        + 1.0879759788513184 * np.tanh((1.1398535966873169 - 0.2350536435842514 * x1 + 0.075554989278316498 * x2))))


def quad_lya(X: np.ndarray) -> np.ndarray:
    P = np.asfarray([
        [ 95106951.78382835,  -27406557.591955528],
        [-27406557.591955528,  92964549.46439281 ]])
    return 0.5*np.sum(np.multiply(X @ P, X), axis=1)
