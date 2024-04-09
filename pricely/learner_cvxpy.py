import cvxpy as cp
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovLearner


class QuadraticLearner(PLyapunovLearner):
    def __init__(self, x_dim: int, u_dim: int = 0, tol: float=10**-6) -> None:
        assert x_dim > 0
        assert tol > 0.0

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._tol = tol
        self._pd_mat = cp.Variable((x_dim, x_dim), name="P", PSD=True)

    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, max_epochs: int=1, **kwargs) -> Sequence[float]:
        # constraints from samples
        yP = y @ self._pd_mat
        yPx = cp.sum(cp.multiply(yP, x), axis=1)
        constraints = [
            self._pd_mat - self._tol >> 0,  # ensure x^T Px > 0 for x != 0
            yPx + cp.sum(x**2, axis=1) <= 0,
        ]

        # The following objective has no justification but works better!
        obj = cp.Minimize(cp.sum_squares(x + 0.5*yP))
        prob = cp.Problem(obj, constraints)

        for solver in [cp.CVXOPT, cp.CLARABEL]:
            try:
                prob.solve(solver)
                break  # Early terminate when a solver suceeded
            except cp.SolverError:
                continue

        if prob.status == "optimal":
            return [obj.value/len(x)]  # type: ignore
        elif prob.status == "infeasible":
            raise RuntimeError("Learner cannot find a quadratic Lyapunov function")
        else:
            raise RuntimeError(f"CVXPY returns status {prob.status}.")

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        return (x_vars @ self._pd_mat.value @ x_vars) / 2.0
    
    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._pd_mat.value) * x_values, axis=1) / 2.0

    def lya_decay_rate(self) -> float:
        # Inverse of Spectral Radius
        eig_max_inv = 1.0 / np.abs(np.linalg.eigvalsh(self._pd_mat.value)).max()
        return eig_max_inv

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        if self._u_dim == 0:
            return []
        else:
            raise NotImplementedError

    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        if self._u_dim == 0:
            return np.array([]).reshape(len(x_values), 0)
        else:
            raise NotImplementedError

    def find_sublevel_set_and_box(self, x_roi: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        level_ub = self.find_level_ub(x_roi)
        abs_x_ub = np.sqrt(2*level_ub * np.linalg.inv(self._pd_mat.value).diagonal())
        return level_ub, abs_x_ub


class SOS1Learner(PLyapunovLearner):
    """ Learning SOS with monomials of degree <= 1 """
    def __init__(self, x_dim: int, u_dim: int = 0, tol: float=10**-6) -> None:
        assert x_dim > 0
        assert tol > 0.0

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._tol = tol
        self._pd_mat = cp.Variable((x_dim + 1, x_dim + 1), name="P", PSD=True)

    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, max_epochs: int=1, **kwargs) -> Sequence[float]:
        # constraints from samples
        yP = y @ self._pd_mat[1:, 1:]
        yPx = cp.sum(cp.multiply(yP, x), axis=1)
        yv = y @ self._pd_mat[1:, 0]
        constraints = [
            self._pd_mat[0, 0] == 0,  # ensure V(0) = 0
            self._pd_mat[1:, 1:] - self._tol >> 0,  # ensure h(x)^T P h(x) > 0 for x != 0
            yv + yPx <= 0.0
        ]

        # Maximize the distance that still ensures yj v + yj P x <= 0
        ## L-2 norm: ||x - xj||^2 <= -2*xj P yj - ||P yj||^2 ==> x P yj <= 0
        ## The above sufficient condition is discovered through the S-procedure.
        ## Let r denote the distance from xj to the hyperplane yj v + yj P x = 0.
        ## Because (P yj/||P yj||) is the normal vector,
        ## we can derive r = -(yj v + xj P yj) / ||P yj|| since yj v +  xj P yj < 0.
        ## Then, -2*(yj v + xj P yj) - ||P yj||^2 = 2r||P yj|| - ||P yj||^2 = r^2 - (r-||P yj||)^2 <= r^2
        obj = cp.Maximize(-2*cp.sum(yv + yPx) - cp.sum_squares(yP))
        ## L-inf norm: dj_max <= -xj P yj / sum(abs(P yj))
        # See "An analytical solution to the minimum Lp-norm of a hyperplane"
        # obj = cp.Maximize(cp.sum(-xPy / cp.sum(Py, axis= 1)))  # XXX Not supported by CVXPY
        prob = cp.Problem(obj, constraints)
        for solver in [cp.CVXOPT, cp.CLARABEL]:
            try:
                prob.solve(solver)
                break  # Early terminate when a solver succeeded
            except cp.SolverError:
                continue
        if prob.status == "optimal":
            return [obj.value/len(x)]  # type: ignore
        elif prob.status == "infeasible":
            raise RuntimeError("Learner cannot find a SOS1 Lyapunov function")
        else:
            raise RuntimeError(f"CVXPY returns status {prob.status}.")

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        z_exprs = [Expr(1.0)]
        z_exprs.extend(x_vars)
        return (z_exprs @ self._pd_mat.value @ z_exprs) / 2.0
    
    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        z_values = np.column_stack((np.ones(shape=(len(x_values), 1)), x_values))
        return np.sum((z_values @ self._pd_mat.value) * z_values, axis=1) / 2.0

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        if self._u_dim == 0:
            return []
        else:
            raise NotImplementedError

    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        if self._u_dim == 0:
            return np.array([]).reshape(len(x_values), 0)
        else:
            raise NotImplementedError
