import os
from typing import Sequence, Tuple

import dreal
import torch
import torch.nn.functional as F
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


class LyapunovNetRegressor(NeuralNetRegressor):
    def __init__(self, module: LyapunovNet, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(module, None, optimizer)

    def train_one_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        assert len(X) == len(y)
        assert X.requires_grad
        x_values, dxdt_values = X, y

        lya_0_value = self._model(torch.zeros_like(x_values[0], device=DEVICE))
        lya_values = self._model(x_values)

        # Explicitly compute values of partial derivative
        """
        p_lya_p_hidden = dtanh(lya_values) @ self._model.layer2.weight
        p_hidden_p_layer1 = dtanh(torch.tanh(
            x_values @ self._model.layer1.weight.t() + self._model.layer1.bias))
        p_layer1_p_x = self._model.layer1.weight
        # Use element-wise multiplication * for the hidden layer because tanh is element-wise
        p_lya_px_values = (p_lya_p_hidden * p_hidden_p_layer1) @ p_layer1_p_x
        """

        # Use PyTorch autograd instead
        p_lya_px_values = torch.autograd.grad(
            lya_values, x_values,
            grad_outputs=torch.ones_like(lya_values),
            retain_graph=True)[0]

        # Lie derivative of V: L_V = ∑_j ∂V(x)/∂x_j*f_j(x)
        lie_der_values = (p_lya_px_values * dxdt_values).sum(dim=1)

        # NOTE Somehow taking mean individually before addition saves CUDA memory
        lya_risk = F.relu(-lya_values).mean() \
            + F.relu(lie_der_values).mean() \
            + lya_0_value.pow(2)
        self.optimizer.zero_grad()
        lya_risk.backward(retain_graph=True)
        self.optimizer.step()

        return lya_risk.item()

    def dreal_expr(self, x_vars: Sequence[dreal.Variable]) -> dreal.Expression:
        # Get weights and biases
        w1 = self.model.layer1.weight.data.cpu().numpy().squeeze()
        w2 = self.model.layer2.weight.data.cpu().numpy().squeeze()
        b1 = self.model.layer1.bias.data.cpu().numpy().squeeze()
        b2 = self.model.layer2.bias.data.cpu().numpy().squeeze()

        z1 = b1 + x_vars @ w1.T
        a1 = [dreal.tanh(z1[j]) for j in range(len(z1))]
        z2 = b2 + a1 @ w2.T
        lya_expr = dreal.tanh(z2)
        return lya_expr


def dtanh(s: torch.Tensor) -> torch.Tensor:
    # Derivative of activation
    return 1.0 - s**2


def gen_equispace_regions(
    x_roi: torch.Tensor,
    x_part: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x_roi.shape == (2, len(x_part))
    x_dim = len(x_part)
    # generate dataset (values of x):
    axes_cuts = (torch.linspace(
        x_roi[0, i], x_roi[1, i], x_part[i]+1,
        dtype=torch.float32) for i in range(x_dim))
    # xx = ((b_i[:-1] + b_i[1:]) / 2 for b_i in bound_pts)
    bound_pts = torch.cartesian_prod(
        *axes_cuts).reshape(tuple(n+1 for n in x_part) + (x_dim,))

    lb_pts = bound_pts[[slice(0, -1)]*x_dim].reshape((-1, x_dim))
    ub_pts = bound_pts[[slice(1, None)]*x_dim].reshape((-1, x_dim))
    x = (lb_pts + ub_pts) / 2
    return x, lb_pts, ub_pts


class KnownLyapunovNet:
    """ Known Lyapunov net from the NeuRIPS 2022 paper """

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def _loss(self, x_values: torch.Tensor, dxdt_values: torch.Tensor) -> float:
        lya_0_value = self.predict(
            torch.zeros_like(x_values[0], device=DEVICE))
        lya_values = self.predict(x_values)
        p_lya_px_values = torch.autograd.grad(
            lya_values, x_values,
            grad_outputs=torch.ones_like(lya_values))[0]
        lie_der_values = (p_lya_px_values * dxdt_values).sum(dim=1)
        lya_risk = F.relu(-lya_values).mean() \
            + F.relu(lie_der_values).mean() \
            + lya_0_value.pow(2)
        return lya_risk.item()

    def fit_loop(self, X: torch.Tensor, y: torch.Tensor, max_epoch: int = 10, copy: bool = True) -> Sequence[float]:
        lya_risk = self._loss(X, y)
        return [lya_risk]*max_epoch

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        linear1 = X @ self.W1.T + self.b1
        hidden1 = torch.tanh(linear1)
        linear2 = hidden1 @ self.W2.T + self.b2
        lya = torch.tanh(linear2)
        return lya

    def dreal_expr(self, x_vars: Sequence[dreal.Variable]) -> dreal.Expression:
        W1 = self.W1.cpu().numpy().squeeze()
        b1 = self.b1.cpu().numpy().squeeze()
        W2 = self.W2.cpu().numpy().squeeze()
        b2 = self.b2.cpu().numpy().squeeze()

        linear1 = x_vars @ W1.T + b1
        hidden1 = [dreal.tanh(expr) for expr in linear1]
        linear2 = hidden1 @ W2.T + b2
        lya_expr = dreal.tanh(linear2)
        return lya_expr
