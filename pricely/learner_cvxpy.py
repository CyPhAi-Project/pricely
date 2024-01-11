import cvxpy as cp
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence

from pricely.cegus_lyapunov import PLyapunovLearner


class QuadraticLearner(PLyapunovLearner):
    def __init__(self, x_dim: int, u_dim: int = 0, tol: float=10**-6) -> None:
        assert x_dim > 0
        assert tol > 0.0

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._tol = tol
        self._pd_mat = cp.Variable((x_dim, x_dim), name="P", PSD=True)

    def fit_loop(self, x: np.ndarray, y: np.ndarray, max_epochs: int=1, **kwargs) -> Sequence[float]:
        # constraints from samples
        Py = y @ self._pd_mat
        xPy = cp.sum(cp.multiply(x, Py), axis=1)
        constraints = [
            cp.abs(self._pd_mat) <= 1.0,
            xPy + self._tol <= 0.0,
        ]

        # Maximize the distance that still ensures yj P x <= 0
        ## L-2 norm: ||x - xj||^2 <= -2*xj P yj - ||P yj||^2 ==> x P yj <= 0
        ## The above sufficient condition is discovered through the S-procedure.
        ## Let r denote the distance from xj to the hyperplane x P yj = 0.
        ## Because the origin is on the hyperplane and (P yj/||P yj||) is the normal vector,
        ## we can derive r = -xj (P yj / ||P yj||) since xj P yj < 0.
        ## Then, -2*xj P yj - ||P yj||^2 = 2r||P yj|| - ||P yj||^2 = r^2 - (r-||P yj||)^2 <= r^2
        obj = cp.Maximize(-cp.sum(2*xPy) - cp.sum_squares(Py))
        ## L-inf norm: dj_max <= -xj P yj / sum(abs(P yj))
        # See "An analytical solution to the minimum Lp-norm of a hyperplane"
        # obj = cp.Maximize(cp.sum(-xPy / cp.sum(Py, axis= 1)))  # XXX Not supported by CVXPY
        prob = cp.Problem(obj, constraints)
        prob.solve()
        if prob.status == "optimal":
            return [obj.value/len(x)]
        elif prob.status == "infeasible":
            raise RuntimeError("Learner cannot find a quadratic Lyapunov function")
        elif prob.status == "optimal_inaccurate":
            prob.solve(solver="SCS", max_iters=2_000_000)
            assert prob.status == "optimal", f"{prob.status}"
            return [obj.value/len(x)]
        else:
            raise RuntimeError(f"CVXPY returns status {prob.status}.")

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        return (x_vars @ self._pd_mat.value @ x_vars) / 2.0
    
    def lya_values(self, x_values: np.ndarray) -> np.ndarray:
        return np.sum((x_values @ self._pd_mat.value) * x_values, axis=1) / 2.0

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        return super().ctrl_exprs(x_vars)

    def ctrl_values(self, x_values: np.ndarray) -> np.ndarray:
        if self._u_dim == 0:
            return np.array([]).reshape(len(x_values), 0)
        else:
            return super().ctrl_values(x_values)
