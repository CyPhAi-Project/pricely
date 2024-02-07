from dreal import atan as ArcTan, Max, Min, sin as Sin, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


THETA_LIM = np.pi / 4  # rad
K_P = 0.45
WHEEL_BASE = 1.75  # m
VEL = 2.8  # m/s
STEER_LIM = 0.61  # rad

X_DIM = 2
X_ROI = np.array([
    [-1.25, -THETA_LIM],  # Lower bounds
    [+1.25, +THETA_LIM]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 0.0625


def ctrl(x: np.ndarray) -> np.ndarray:
    u = np.zeros(shape=(len(x), 1))
    u[:, 0] = np.clip(x[:, 1] + np.arctan(K_P*x[:, 0] / VEL),
                      a_min=-STEER_LIM, a_max=STEER_LIM)
    return u


def f_bbox(x: np.ndarray) -> np.ndarray:
    u = ctrl(x)
    dxdt = np.zeros_like(x)
    dxdt[:, 0] = VEL*np.sin(x[:, 1] - u[:, 0])
    dxdt[:, 1] = -(VEL/WHEEL_BASE)*np.sin(u[:, 0])
    return dxdt


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    lip_ub = np.sqrt((1+1/WHEEL_BASE**2)*(VEL**2 + K_P**2))  # Manually derived
    return np.full(len(x_regions), fill_value=lip_ub)


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    u_expr = x_vars[1] + ArcTan(K_P*x_vars[0] / VEL)
    u_clip_expr = Min(Max(u_expr, -STEER_LIM), STEER_LIM)
    dxdt_exprs = [
        VEL*Sin(x_vars[1] - u_clip_expr),
        -(VEL/WHEEL_BASE)*Sin(u_clip_expr)
    ]
    return dxdt_exprs
