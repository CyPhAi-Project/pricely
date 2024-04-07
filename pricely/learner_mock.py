from dreal import Expression as Expr, Variable  #type: ignore
import numpy as np
from typing import Optional, Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovLearner


class MockQuadraticLearner(PLyapunovLearner):
    def __init__(self, pd_mat: np.ndarray, ctrl_mat: Optional[np.ndarray] = None) -> None:
        assert len(pd_mat.shape) == 2 and pd_mat.shape[0] == pd_mat.shape[1]
        assert (pd_mat==pd_mat.T).all()
        assert np.all(np.linalg.eigvalsh(pd_mat) >= 0.0)

        self._pd_mat = pd_mat
        if ctrl_mat is None:
            self._ctrl_mat = np.array([]).reshape(pd_mat.shape[0], 0)
        else:
            self._ctrl_mat = ctrl_mat

    def fit_loop(self, X: NDArrayFloat, y: NDArrayFloat, **kwargs):
        return [0.0]
    
    def lya_expr(self, x_vars: Sequence) -> Expr:
        return (x_vars @ self._pd_mat @ x_vars) / 2.0

    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._pd_mat) * x_values, axis=1) / 2.0
    
    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        if self._ctrl_mat.shape[1] == 0:
            return []
        raise NotImplementedError

    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        if self._ctrl_mat.shape[1] == 0:
            return np.array([]).reshape(len(x_values), 0)
        raise NotImplementedError

    def find_sublevel_set_and_box(self, x_roi: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        level_ub = self.find_level_ub(x_roi)
        abs_x_ub = np.sqrt(2*level_ub * np.linalg.inv(self._pd_mat).diagonal())
        return level_ub, abs_x_ub
