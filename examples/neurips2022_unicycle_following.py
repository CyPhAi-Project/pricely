from dreal import cos as Cos, sin as Sin, tanh as Tanh, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence

KAPPA = 1.0
VEL = 1.0  # m/s

X_DIM = 2
X_NORM_LB = 0.1
X_NORM_UB = 0.8
X_LIM = np.array([
    [-X_NORM_UB]*X_DIM, # Lower bounds
    [+X_NORM_UB]*X_DIM  # Upper bounds
])
assert X_LIM.shape == (2, X_DIM)
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


def nnet_lya(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.tanh(
        (0.59095233678817749 
        - 1.2369513511657715 * np.tanh((-1.9072647094726562 - 2.1378724575042725 * x1 + 1.0794919729232788 * x2))
        + 1.4756215810775757 * np.tanh((-0.79773855209350586 + 4.9804959297180176 * x1 + 0.11680498719215393 * x2))
        - 2.1383264064788818 * np.tanh((0.18891614675521851 + 2.8365912437438965 * x1 + 0.69793730974197388 * x2))
        - 0.76876986026763916 * np.tanh((0.73854517936706543 - 3.338552713394165 * x1 - 2.2363924980163574 * x2))
        + 1.0839570760726929 * np.tanh((0.87543833255767822 - 0.027711296454071999 * x1 - 0.25035503506660461 * x2))
        - 0.84737318754196167 * np.tanh((1.0984361171722412 + 0.61321312189102173 * x1 - 1.6286146640777588 * x2))))


def quad_lya(X: np.ndarray) -> np.ndarray:
    P = np.asfarray([
        [96993655.36465222,  20809870.734031927],
        [20809870.734031927, 16434242.326128198]])
    return 0.5*np.sum(np.multiply(X @ P, X), axis=1)
