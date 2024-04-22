from dreal import Expression as Expr, Variable  #type: ignore
import numpy as np
from typing import Optional, Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayFloat, PLyapunovCandidate, PLyapunovLearner
from pricely.candidates import QuadraticLyapunov


class MockQuadraticLearner(PLyapunovLearner):
    """
    A mocker learner always proposing the same Quadratic Lyapunov function and linear controller.
    """
    def __init__(self, pd_mat: np.ndarray, ctrl_mat: Optional[np.ndarray] = None) -> None:
        assert len(pd_mat.shape) == 2 and pd_mat.shape[0] == pd_mat.shape[1]
        assert (pd_mat==pd_mat.T).all()
        assert np.all(np.linalg.eigvalsh(pd_mat) >= 0.0)

        self._pd_mat = pd_mat
        if ctrl_mat is None:
            self._ctrl_mat = np.array([]).reshape(0, pd_mat.shape[0])
        else:
            assert ctrl_mat.shape[1] == pd_mat.shape[0]
            self._ctrl_mat = ctrl_mat

    @property
    def x_dim(self) -> int:
        return self._pd_mat.shape[0]

    @property
    def u_dim(self) -> int:
        return self._ctrl_mat.shape[0]

    def fit_loop(self, x: NDArrayFloat, u: NDArrayFloat, y: NDArrayFloat, **kwargs):
        return [0.0]

    def get_candidate(self) -> PLyapunovCandidate:
        return QuadraticLyapunov(self._pd_mat, self._ctrl_mat, 0.0)
