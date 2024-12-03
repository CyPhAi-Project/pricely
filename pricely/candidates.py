from dreal import Expression as Expr, Variable  # type: ignore
import numpy as np
from typing import Optional, Sequence

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovCandidate


class QuadraticLyapunov(PLyapunovCandidate):
    def __init__(self, sym_mat: NDArrayFloat, ctrl_mat: Optional[np.ndarray] = None, decay_rate: float=0.0) -> None:
        assert sym_mat.ndim == 2 and sym_mat.shape[0] == sym_mat.shape[1]
        assert np.isfinite(decay_rate) and decay_rate >= 0.0
        self._sym_mat = sym_mat
        self._lambda = decay_rate

        if ctrl_mat is None:
            self._ctrl_mat = np.array([]).reshape(0, sym_mat.shape[0])
        else:
            assert ctrl_mat.shape[1] == sym_mat.shape[0]
            self._ctrl_mat = ctrl_mat

    def __copy__(self):
        return QuadraticLyapunov(
            sym_mat=self._sym_mat.copy(),
            ctrl_mat=self._ctrl_mat.copy(),
            decay_rate=self._lambda)

    def __str__(self) -> str:
        with np.printoptions(floatmode='unique'):
            return "V(x) = ½ x·(Ax) with A =\n" + str(self._sym_mat)

    def u_dim(self) -> int:
        return self._ctrl_mat.shape[0]

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expr:
        return (x_vars @ self._sym_mat @ x_vars) / 2.0

    def lya_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._sym_mat) * x_values, axis=1) / 2.0

    def lie_der_values(self, x_values: NDArrayFloat, y_values: NDArrayFloat) -> NDArrayFloat:
        return np.sum((x_values @ self._sym_mat) * y_values, axis=1)

    def lya_decay_rate(self) -> float:
        # Inverse of Spectral Radius
        eig_max_inv = self._lambda / np.abs(np.linalg.eigvalsh(self._sym_mat)).max()
        return eig_max_inv

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expr]:
        return [] if self.u_dim == 0 \
            else (self._ctrl_mat @ x_vars).tolist()

    def ctrl_values(self, x_values: NDArrayFloat) -> NDArrayFloat:
        return np.array([]).reshape(len(x_values), 0) if self.u_dim == 0 \
            else x_values @ self._ctrl_mat.T
