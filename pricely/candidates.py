from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Optional, Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovCandidate


class QuadraticLyapunov(PLyapunovCandidate):
    def __init__(self, pd_mat: NDArrayFloat, ctrl_mat: Optional[np.ndarray] = None, decay_rate: float=0.0) -> None:
        assert pd_mat.ndim == 2 and pd_mat.shape[0] == pd_mat.shape[1]
        assert np.isfinite(decay_rate) and decay_rate >= 0.0
        self._pd_mat = pd_mat
        self._lambda = decay_rate

        if ctrl_mat is None:
            self._ctrl_mat = np.array([]).reshape(0, pd_mat.shape[0])
        else:
            assert ctrl_mat.shape[1] == pd_mat.shape[0]
            self._ctrl_mat = ctrl_mat

    def u_dim(self) -> int:
        return self._ctrl_mat.shape[0]

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        return (x_vars @ self._pd_mat @ x_vars) / 2.0

    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._pd_mat) * x_values, axis=1) / 2.0

    def lie_der_values(self, x_values: NDArrayFloat, y_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._pd_mat) * y_values, axis=1)

    def lya_decay_rate(self) -> float:
        # Inverse of Spectral Radius
        eig_max_inv = self._lambda / np.abs(np.linalg.eigvalsh(self._pd_mat)).max()
        return eig_max_inv

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        return [] if self.u_dim == 0 \
            else (self._ctrl_mat @ x_vars).tolist()

    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.array([]).reshape(len(x_values), 0) if self.u_dim == 0 \
            else x_values @ self._ctrl_mat.T

    def find_sublevel_set_and_box(self, x_roi: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        level_ub = self.find_level_ub(x_roi)
        abs_x_ub = np.sqrt(2*level_ub * np.linalg.inv(self._pd_mat).diagonal())
        return level_ub, abs_x_ub