"""
ODE from "A novel trajectory-tracking control law for wheeled mobile robots"
"""
from dreal import Expression as Expr, cos as Cos, sin as Sin, Variable  # type: ignore
import numpy as np
from typing import Sequence


X_DIM = 3
X_ROI = np.array([
    [-0.75, -0.75, -np.pi/4], # Lower bounds
    [+0.75, +0.75, +np.pi/4]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 2**-8

KNOWN_QUAD_LYA = np.eye(X_DIM, X_DIM)

v_r = 1.0  # m/s
w_r = 2**-4  # rad
K = 1.0  # Require k > 0
K_x = 1.0
K_s = 4.0
a = 4.0  # Require a > 2
n = 0  # n in [-2, -1, 0, 1, 2]


def ctrl(err: np.ndarray) -> np.ndarray:
    e_x, e_y, e_theta = err[:, 0], err[:, 1], err[:, 2]
    e_s, e_c = np.sin(e_theta), 1.0 - np.cos(e_theta)
    u = np.zeros(shape=(len(err), 2))
    u[:, 0] = K_x*e_x
    u[:, 1] = K*v_r*e_y*(1 + e_c/a)**2 + K_s*e_s*(1 + e_c/a)**(2*n)
    return u


def dyn(err: np.ndarray, u: np.ndarray) -> np.ndarray:
    e_x, e_y, e_theta = err[:, 0], err[:, 1], err[:, 2]
    v_b, w_b = u[:, 0], u[:, 1]
    d_err = np.zeros_like(err)
    d_err[:, 0] = w_r*e_y - v_b + e_y*w_b
    d_err[:, 1] = -w_r*e_x + v_r*np.sin(e_theta) - e_x*w_b
    d_err[:, 2] = -w_b
    return d_err


def f_bbox(err: np.ndarray) -> np.ndarray:
    dx = dyn(err, ctrl(err))
    return dx


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    We further upper bound each state variable with its absolute value.
    """
    abs_furtherest = np.max(np.abs(x_regions), axis=1)
    abs_x, abs_y, _ = abs_furtherest[:, 0], abs_furtherest[:, 1], abs_furtherest[:, 2]

    w_b_ub = K*v_r*abs_y*(1 + 2/a)**2 + K_s
    dw_b_dy_ub = K*v_r*(1 + 2/a)**2
    if n != 0:
        raise NotImplementedError("TODO")
    else:
        dw_b_dtheta_ub = K*v_r*abs_y*2*(1 + 2/a)*(1/a) + K_s

    res = np.sqrt(
        K_x**2 + (w_r + w_b_ub + abs_y*dw_b_dy_ub)**2 + (abs_y*dw_b_dtheta_ub)**2 +
        w_b_ub**2 + (abs_x*dw_b_dy_ub)**2 + (v_r + abs_x*dw_b_dtheta_ub)**2 +
        0 + dw_b_dy_ub**2 + dw_b_dtheta_ub**2
    ).squeeze()
    return res


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    e_x, e_y, e_theta = x_vars
    e_s, e_c = Sin(e_theta), 1.0 - Cos(e_theta)
    u_exprs = [
        K_x*e_x,
        K*v_r*e_y*(1 + e_c/a)**2 + K_s*e_s*(1 + e_c/a)**(2*n)
    ]
    v_b, w_b = u_exprs
    return [
        w_r*e_y - v_b + e_y*w_b,
        -w_r*e_x + v_r*e_s - e_x*w_b,
        -w_b]

def test_lip():
    x_vars = [Variable(s) for s in "xyθ"]

    for exp in f_expr(x_vars):
        print("\n")
        for x in x_vars:
            print(exp.Differentiate(x))


if __name__ == "__main__":
    test_lip()
