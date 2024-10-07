import cvxpy as cp
from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Literal, Sequence

from pricely.candidates import QuadraticLyapunov
from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovLearner


class QuadraticLearner(PLyapunovLearner):
    def __init__(self, x_dim: int, u_dim: int = 0, tol: float=10**-6,
                v_max: float= 10**3,
                method:Literal["analytic", "chebyshev", "volumetric"] = "analytic") -> None:
        assert x_dim > 0
        assert tol > 0.0
        assert v_max > 0.0

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._tol = tol
        self._v_max = v_max
        self._build_prob = {
            "analytic": self._analytic,
            "chebyshev": self._chebyshev,
            "volumetric": self._volumetric,
        }[method]
        self._pd_mat = cp.Variable((x_dim, x_dim), name="P", PSD=True)
        # FIXME: Need to pick different objectives in order to support exponential stability
        self._lambda = cp.Parameter(name="λ", value=0.0)  # XXX: Change to a variable for exponential stability

    def _analytic(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> cp.Problem:
        """
        Find a good candidate in the polytope by choosing the analytic center
        See Section 4.1 in "Learning control lyapunov functions from counterexamples and demonstrations"
        """
        yP = y @ self._pd_mat
        yPx = cp.sum(cp.multiply(yP, x), axis=1)
        # Find the analytic center
        constraints = [
            self._pd_mat <= self._v_max,
            self._pd_mat >= -self._v_max,
            self._pd_mat >> 0,  # ensure x^T Px >= 0
            yPx <= 0,
        ]
        obj = cp.Maximize(
            cp.sum(cp.log(self._v_max - self._pd_mat)) +
            cp.sum(cp.log(self._v_max + self._pd_mat)) + 
            cp.sum(cp.log(-yPx)))
        return cp.Problem(obj, constraints)

    def _chebyshev(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> cp.Problem:
        """
        Find a candidate in the polytope defined by choosing the Chebyshev center
        """
        raise NotImplementedError("Using Chebyshev center is not supported yet.")

    def _volumetric(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> cp.Problem:
        """
        Find a candidate in the polytope defined by choosing the Chebyshev center
        """
        raise NotImplementedError("Using volumetric center is not supported yet.")

    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, max_epochs: int=1, **kwargs) -> Sequence[float]:
        prob = self._build_prob(x, u, y)

        for solver in [cp.CLARABEL, cp.SCS]:
            try:
                prob.solve(solver)
                break  # Early terminate when a solver suceeded
            except cp.SolverError:
                continue

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return [prob.objective.value/len(x)]  # type: ignore
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise RuntimeError("Learner cannot find a quadratic Lyapunov function")
        else:
            raise RuntimeError(f"CVXPY returns status {prob.status}.")

    def get_candidate(self) -> QuadraticLyapunov:
        assert self._lambda.value is not None

        if self._u_dim == 0:
            if self._pd_mat.value is None:
                # Initial candidate is a zero matrix
                return QuadraticLyapunov(np.zeros(shape=(self._x_dim, self._x_dim)), decay_rate=self._lambda.value)
            return QuadraticLyapunov(self._pd_mat.value, decay_rate=self._lambda.value)
        else:
            raise NotImplementedError("Learning Lyapunov controller is not supported yet.")
