from dreal import cos as Cos, sin as Sin, tanh as Tanh, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence

KAPPA = 1.0
VEL = 1.0  # m/s

X_DIM = 2
THETA_LIM = np.pi / 4
X_ROI = np.array([
    [-0.75, -THETA_LIM],  # Lower bounds
    [+0.75, +THETA_LIM]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 2**-6

B_MAT = np.array([
    [2.75, 0],
    [0.625, 1]])
KNOWN_QUAD_LYA = B_MAT.T @ B_MAT


def ctrl(x: np.ndarray) -> np.ndarray:
    d_e, theta_e = x[:, 0], x[:, 1]
    return 5.0*np.tanh(
        - 5.95539*d_e
        - 4.03426*theta_e
        + 0.19740).reshape(len(x), 1)

def dyn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    d_e, theta_e = x[:, 0], x[:, 1]
    w = u[:, 0]
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = VEL*np.sin(theta_e)
    dxdt[:, 1] = w - VEL*KAPPA*np.cos(theta_e)/(1-d_e*KAPPA)
    return dxdt

def f_bbox(x: np.ndarray) -> np.ndarray:
    assert x.shape[1] == X_DIM
    return dyn(x, ctrl(x))


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    # Avoid singular states when K*d == 1
    d_ubs = x_regions[:, 2, 0]
    assert np.all(KAPPA*d_ubs < 1.0)
    tmp = (1.0 - KAPPA*d_ubs)
    lip_ubs = np.sqrt(
        VEL**2 + 
        (5.0*-5.95539 - VEL*KAPPA*KAPPA/(tmp**2))**2 + 
        2*(5.0*-4.03426)**2 + 2*(VEL*KAPPA/tmp)**2)
    return lip_ubs


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    d_e, theta_e = x_vars
    w_expr = 5.0*Tanh(
        - 5.95539*d_e
        - 4.03426*theta_e
        + 0.19740)
    return [VEL*Sin(theta_e),
            w_expr - VEL*KAPPA*Cos(theta_e)/(1-d_e*KAPPA)]
