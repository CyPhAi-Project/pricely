from dreal import sin as Sin, tanh as Tanh, Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence


def known_lya_values(x_values: np.ndarray) -> np.ndarray:
    W1_T = np.asfarray([
        [+0.03331, +0.03467, +2.12564, -0.39925, +0.12885, +0.95375],
        [-0.03113, -0.01892, +0.02354, -0.10678, -0.32245, +0.01298]
    ])
    b1 = np.asfarray([-0.48061, 0.88048, 0.86448, -0.87253, 0.81866, -0.26619])
    W2 = np.asfarray([
        [-0.33862, 0.65177, -0.52607, 0.23062, -0.04802, 0.66825]
    ])
    b2 = np.asfarray([0.22032])
    linear1 = x_values @ W1_T + b1
    hidden1 = np.tanh(linear1)
    linear2 = hidden1 @ W2.T + b2
    return np.tanh(linear2).squeeze()


G = 9.81  # m/s^2
M = 0.15  # kg
L = 0.5  # m

THETA_LIM = np.pi / 4

X_ROI = np.array([
    [-THETA_LIM, -2],  # Lower bounds
    [+THETA_LIM, +2]  # Upper bounds
])
X_DIM = 2
ABS_X_LB = 0.0625
LIP_CAP = 10000.0  # Ignore regions with Lipschitz constant exceed this cap

B_MAT = np.array([
    [5, 0],
    [1.25, 0.25]])
KNOWN_QUAD_LYA = B_MAT.T @ B_MAT


def ctrl(x: np.ndarray) -> np.ndarray:
    theta, omega = x[:, 0], x[:, 1]
    return 20.0*np.tanh(
        -23.28632*theta
        -5.27055*omega).reshape(len(x),)

def dyn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    theta, omega = x[:, 0], x[:, 1]

    dxdt = np.zeros_like(x)
    dxdt[:, 0] = omega
    dxdt[:, 1] = (M*G*L*np.sin(theta) + u - 0.1*omega) / (M*L**2)
    return dxdt

def f_bbox(x: np.ndarray) -> np.ndarray:
    assert x.shape[1] == X_DIM
    return dyn(x, ctrl(x))


def f_expr(x_vars: Sequence[Variable]) -> Sequence[Expr]:
    assert len(x_vars) == X_DIM
    theta, omega = x_vars
    u_expr = 20.0*Tanh(-23.28632*theta - 5.27055*omega)
    return [omega,
            (M*G*L*Sin(theta) + u_expr - 0.1*omega) / (M*L**2)]


def calc_lip_bbox(x_regions: np.ndarray) -> np.ndarray:
    """
    Use Frobenious Norm of the Jacobian matrix to provide a local Lipschitz constant
    The supremum is at the point that minimizes tanh(-23.28632*theta - 5.27055*omega);
    hence, we can pick the point closest to the line: -23.28632*theta - 5.27055*omega == 0.
    We further use the vertice to check if the hyperrectangle contains a point on the line,
    otherwise we can use the closest vertex.
    """
    assert x_regions.shape[1] == 3 and x_regions.shape[2] == X_DIM
    values = -23.28632*x_regions[:, :, 0] - 5.27055*x_regions[:, :, 1]
    assert values.shape == (len(x_regions), 3)
    min_values = np.where(np.sign(values[:, 1]) == np.sign(values[:, 2]),
                          np.min(np.abs(values), axis=1), 0.0)
    assert min_values.ndim == 1 and len(min_values) == len(x_regions)
    max_du = 1.0 - np.tanh(min_values)**2
    # Forbenious norm
    # lip_ub = np.sqrt(((M*L**2)**2 + (20*5.27055*max_du - 0.1)**2 + (M*G*L + 20*23.28632*max_du)**2)) / (M*L**2)
    # Convex upper bound for the Forbenious norm
    lip_ub = np.sqrt(((M*L**2)**2 + (20*5.27055*max_du)**2 + 0.1**2 + (M*G*L + 20*23.28632*max_du)**2)) / (M*L**2)
    return np.full(len(x_regions), fill_value=lip_ub)
