from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence

X_DIM = 1

X_ROI = np.array([
    [-2.0],  # Lower bounds
    [+2.0]  # Upper bounds
])
assert X_ROI.shape == (2, X_DIM)
ABS_X_LB = 2**-6
KNOWN_QUAD_LYA = np.eye(1)

K = 0.125
def f_bbox(x: np.ndarray) -> np.ndarray:
    assert x.shape[1] == X_DIM
    return -K*(2*x**3 - 4*x**2 + 3*x)


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    x = x_vars[0]
    return [-K*(2*x**3 - 4*x**2 + 3*x)]


def calc_lip_bbox(x_regions: np.ndarray, global_lip: bool = False) -> np.ndarray:
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    if global_lip:
        return np.full_like(x_regions[:, 0], 5.375).squeeze()

    def jacob(x: np.ndarray) -> np.ndarray:
        return K*np.abs(6*x**2 - 8*x + 3)

    x_lbs, x_ubs = x_regions[:, 1], x_regions[:, 2]
    return np.maximum(jacob(x_lbs), jacob(x_ubs)).squeeze()
