import itertools
import cvxpy as cp
from dreal import Expression as Expr, Variable  # type: ignore
import math
import numpy as np
import torch
from typing import Sequence

from cegus_lyapunov import PLyapunovLearner
from nnet_utils import DEVICE


class QuadraticLearner(PLyapunovLearner):
    def __init__(self, x_dim: int, u_dim: int = 0, tol: float=10**-6) -> None:
        assert x_dim > 0
        assert tol > 0.0

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._tol = tol
        self._pd_mat = cp.Variable((x_dim, x_dim), name="P", PSD=True)

    def fit_loop(self, X: torch.Tensor, y: torch.Tensor, max_epochs: int=1, **kwargs) -> Sequence[float]:
        x_np = X.cpu().detach().numpy().squeeze()
        y_np = y.cpu().detach().numpy().squeeze()

        # constraints from samples
        Py = y_np @ self._pd_mat
        xPy = cp.sum(cp.multiply(x_np, Py), axis=1)
        constraints = [
            xPy + self._tol <= 0
        ]

        # Maximize the distance that still ensures yj P x <= 0
        ## L-2 norm:  ||x - xj||^2 <= -2*xj P yj - ||P yj||^2 ==> x P yj <= 0
        ## The above sufficient condition is discovered through the S-procedure.
        obj = cp.Maximize(cp.sum(-2*xPy - cp.norm2(Py, axis=1)**2))
        ## L-inf norm: dj_max <= -xj P yj / sum(abs(P yj))
        # See "An analytical solution to the minimum Lp-norm of a hyperplane"
        # obj = cp.Maximize(cp.sum(-xPy / cp.sum(Py, axis= 1)))  # XXX Not supported
        prob = cp.Problem(obj, constraints)
        prob.solve()
        if prob.status == "optimal":
            return [obj.value/len(x_np)]
        elif prob.status == "infeasible":
            raise RuntimeError("Learner cannot find a quadratic Lyapunov function")
        else:
            raise RuntimeError(f"CVXPY returns status {prob.status}.")

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        return (x_vars @ self._pd_mat.value @ x_vars) / 2.0
    
    def lya_values(self, x_values: torch.Tensor) -> torch.Tensor:
        mat = torch.tensor(self._pd_mat.value, device=DEVICE)
        return torch.sum((x_values @ mat) * x_values, dim=1) / 2.0

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        return super().ctrl_exprs(x_vars)

    def ctrl_values(self, x_values: torch.Tensor) -> torch.Tensor:
        if self._u_dim == 0:
            return torch.tensor([], device=DEVICE).reshape(len(x_values), 0)
        else:
            return super().ctrl_values(x_values)


class SOSLearner(PLyapunovLearner):
    def __init__(self, x_dim: int, u_dim: int = 0, max_deg: int = 2, tol: float=10**-6) -> None:
        raise NotImplementedError("SOSLearner implementaion is not finished.")
        assert x_dim > 0
        assert tol > 0.0
        assert 2 <= max_deg

        self._u_dim = u_dim
        self._tol = tol

        self._z_dim = math.comb(x_dim+max_deg, max_deg)
        self._pd_mat = cp.Variable((self._z_dim, self._z_dim), name="Q", PSD=True)

        self._to_z_exprs = lambda x_vars: self.x_to_z_expr(x_vars, x_dim, max_deg)
        self._to_z_values = lambda x_values: self.x_to_z_val(x_values, x_dim, max_deg)

    @staticmethod
    def x_to_z_expr(x_vars: Sequence[Variable], x_dim: int, max_deg: int) -> Sequence[Expr]:
        x_vars = np.asarray(x_vars)

        exprs = [Expr(1.0)]
        exprs.extend(x_vars)
        for d in range(2, max_deg+1):
            for indices in itertools.combinations_with_replacement(range(0, x_dim), d):
                expr = np.prod(x_vars[indices,])
                exprs.append(expr)
        return exprs

    @staticmethod
    def x_to_z_val(x_values: torch.Tensor, x_dim: int, max_deg: int) -> torch.Tensor:
        arrs = [
            torch.ones(size=(len(x_values), 1), device=DEVICE),
            x_values
        ]
        for d in range(2, max_deg+1):
            for indices in torch.combinations(torch.arange(0, x_dim), d, with_replacement=True):
                monom_val = torch.prod(x_values[:, indices], dim=1, keepdim=True)
                arrs.append(monom_val)
        assert all(arr.shape[0] == len(x_values) for arr in arrs)
        return torch.column_stack(arrs)


    def fit_loop(self, X: torch.Tensor, y: torch.Tensor, max_epochs: int=1, **kwargs) -> Sequence[float]:
        z_in = self._to_z_values(X)
        z_out = self._to_z_values(y)
        z_in_np = z_in.cpu().detach().numpy().squeeze()
        z_out_np = z_out.cpu().detach().numpy().squeeze()

        lhs = cp.sum(cp.multiply((z_in_np @ self._pd_mat), z_out_np), axis=1)
        constraints = [
            self._pd_mat[0, 0] == 0.0,  # Ensure V(0) == 0
            lhs + self._tol <= 0.0
        ]

        # Minimize the sum of the perimeters for the regions that are impossible to prove
        obj = cp.Minimize(cp.max(lhs))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        if prob.status == "optimal":
            return [obj.value]
        elif prob.status == "infeasible":
            raise RuntimeError("Learner cannot find a SOS Lyapunov function")
        else:
            assert prob.status == "unbounded"
            raise RuntimeError("CVXPY returns unbounded.")
  
    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        z_exprs = self._to_z_exprs(x_vars)
        return (z_exprs @ self._pd_mat.value @ z_exprs)
 
    def lya_values(self, x_values: torch.Tensor) -> torch.Tensor:
        mat = torch.tensor(self._pd_mat.value, device=DEVICE)
        z_values = self._to_z_values(x_values)
        return torch.sum((z_values @ mat) * z_values, dim=1)

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        return super().ctrl_exprs(x_vars)

    def ctrl_values(self, x_values: torch.Tensor) -> torch.Tensor:
        if self._u_dim == 0:
            return torch.tensor([], device=DEVICE).reshape(len(x_values), 0)
        else:
            return super().ctrl_values(x_values)
