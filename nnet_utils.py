import os
from typing import Sequence, Tuple

import torch
import tqdm


# Ensure deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

CPU = "cpu"
# If cuda is available, default using the last GPU of all GPUs
DEVICE = torch.device(
    f'cuda:{torch.cuda.device_count()-1}' if torch.cuda.is_available() else CPU)


class DynamicsNet(torch.nn.Module):
    """ NN for learning dynamics """

    def __init__(self, n_input, n_hidden1, n_output):
        super().__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden1)
        self.layer2 = torch.nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return out


class LyapunovNet(torch.nn.Module):
    """ Neural network model for Lyapunov function """

    def __init__(self, n_input, n_hidden):
        super(LyapunovNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        return out


class NeuralNetRegressor:
    """
    Wrapper class for PyTorch following scikit-learn style API.

    Inspired by `skorch` package. Skorch is however too slow for unknown reasons.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        criterion: torch.nn.MSELoss,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Model is assumed to be always on the device.
        self._model = module.to(DEVICE)
        self._criterion = criterion
        self._optimizer = optimizer

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer

    def train_one_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        assert X.device == DEVICE, f'Tensor X is on {X.device} not on "{DEVICE}"'
        assert y.device == DEVICE, f'Tensor y is on {y.device} not on "{DEVICE}"'

        y_nn = self._model(X)
        loss = self._criterion(y_nn, y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def fit_loop(self, X: torch.Tensor, y: torch.Tensor, max_epoch: int = 10, copy: bool = True) -> Sequence[float]:
        if copy:  # Make a copy on the device.
            X_device = X.to(DEVICE, copy=True)
            y_device = y.to(DEVICE, copy=True)
        else:
            assert X.device == DEVICE, f'Tensor X is on {X.device} not on "{DEVICE}"'
            assert y.device == DEVICE, f'Tensor y is on {y.device} not on "{DEVICE}"'
            X_device = X
            y_device = y

        self._model.train(True)
        losses = []
        for i in tqdm.tqdm(range(max_epoch), ascii=True):
            loss = self.train_one_step(X_device, y_device)
            losses.append(loss)

        if copy:
            del X_device
            del y_device
        torch.cuda.empty_cache()
        return losses

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X, False)

    def predict(self, X: torch.Tensor, copy: bool = True) -> torch.Tensor:
        if copy:  # Make a copy on the device.
            X_device = X.to(DEVICE, copy=True)
        else:
            assert X.device == DEVICE, f'Tensor X is on {X.device} not on "{DEVICE}"'
            X_device = X
        self._model.train(False)
        y = self._model(X_device)
        if copy:
            del X_device
        torch.cuda.empty_cache()
        return y


def gen_equispace_regions(
    part: Sequence[int],
    x_roi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x_roi.shape[0] == 2
    x_dim = x_roi.shape[1]
    assert len(part) == x_dim

    # generate dataset (values of x):
    x_axes_cuts = (torch.linspace(
        x_roi[0, i], x_roi[1, i], part[i]+1,
        dtype=torch.float32) for i in range(x_dim))
    # xx = ((b_i[:-1] + b_i[1:]) / 2 for b_i in bound_pts)
    bound_pts = torch.cartesian_prod(*x_axes_cuts)
    bound_pts.resize_(tuple(n+1 for n in part) + (len(part),))

    lb_pts = bound_pts[[slice(0, -1)]*len(part)].reshape((-1, len(part)))
    ub_pts = bound_pts[[slice(1, None)]*len(part)].reshape((-1, len(part)))
    x = (lb_pts + ub_pts) / 2
    return x, lb_pts, ub_pts
