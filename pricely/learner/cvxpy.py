import cvxpy as cp
import numpy as np
from typing import List, Literal, Sequence, Tuple

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
        self._eps = cp.Variable(1, name="ϵ", nonneg=True)
        self._v_max = v_max
        self._build_prob = {
            "analytic": self._analytic,
            "chebyshev": self._chebyshev,
            "volumetric": self._volumetric,
        }[method]
        self._sym_mat = cp.Variable((x_dim, x_dim), name="P", symmetric=True)
        # FIXME: Need to pick different objectives in order to support exponential stability
        self._lambda = cp.Parameter(name="λ", value=0.0)  # XXX: Change to a variable for exponential stability

    def _analytic(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> Tuple[cp.Maximize, List[cp.Constraint]]:
        """
        Find a good candidate in the polytope by choosing the analytic center
        See Section 4.1 in "Learning control lyapunov functions from counterexamples and demonstrations"
        """
        xP = x @ self._sym_mat
        xPx = cp.sum(cp.multiply(xP, x), axis=1)
        yP = y @ self._sym_mat
        yPx = cp.sum(cp.multiply(yP, x), axis=1)
        # Find the analytic center
        constraints = [
            self._sym_mat >= -self._v_max,
            self._sym_mat <= self._v_max,
            xPx >= self._eps,  # g_x(P) < 0 in paper
            yPx <= -self._eps,  # h_x(P) < 0 in paper
            self._eps <= 2**10*self._tol
        ]
        obj = cp.Maximize(
            cp.sum(cp.log(self._v_max - self._sym_mat)) + cp.sum(cp.log(self._v_max + self._sym_mat)) +
            cp.sum(cp.log(xPx)) + cp.sum(cp.log(-yPx)))
        return obj, constraints  # type: ignore

    def _chebyshev(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> Tuple[cp.Maximize, List[cp.Constraint]]:
        """
        Find a candidate in the polytope defined by choosing the Chebyshev center
        """
        raise NotImplementedError("Using Chebyshev center is not supported yet.")

    def _volumetric(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat) -> Tuple[cp.Maximize, List[cp.Constraint]]:
        """
        Find a candidate in the polytope defined by choosing the Chebyshev center
        """
        raise NotImplementedError("Using volumetric center is not supported yet.")

    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, max_epochs: int=1, **kwargs) -> Sequence[float]:
        obj, cons = self._build_prob(x, u, y)

        FEASIBILITY_SOLVERS = [cp.CVXOPT, cp.CLARABEL]

        fail_count = 0
        for solver in FEASIBILITY_SOLVERS:
            try:
                # Check feasibility first
                feasibility = cp.Problem(cp.Maximize(self._eps), cons)
                feasibility.solve(solver)
                if feasibility.status in [cp.INFEASIBLE]:
                    return []
                elif feasibility.status in [cp.OPTIMAL]:
                    if feasibility.objective.value <= self._tol:  #type: ignore
                        return []  # Almost infeasible
                    # else:
                    break
                else:
                    raise RuntimeError(f"Unexpected status for feasibility check {feasibility.status}")
            except cp.SolverError:
                fail_count += 1
        if fail_count == len(FEASIBILITY_SOLVERS):
            raise RuntimeError(f"All CVXPY solvers have failed in feasbility check.")

        for solver in [cp.CLARABEL, cp.SCS]:
            try:
                # Optimize the logarithmic barrier objective
                prob = cp.Problem(obj)
                prob.solve(solver)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT]:
                    assert not np.any(np.isnan(self._sym_mat.value))
                    return [prob.objective.value/len(x)]  # type: ignore
                else:
                    raise RuntimeError(f"Unexpected status for optimization {prob.status}")
            except cp.SolverError:
                pass
        raise RuntimeError(f"All CVXPY solvers have failed in optimization.")

    def get_candidate(self) -> QuadraticLyapunov:
        assert self._lambda.value is not None

        if self._u_dim == 0:
            if self._sym_mat.value is None:
                raise AttributeError("No candidate available.")
            return QuadraticLyapunov(self._sym_mat.value, decay_rate=self._lambda.value)
        else:
            raise NotImplementedError("Learning Lyapunov controller is not supported yet.")
