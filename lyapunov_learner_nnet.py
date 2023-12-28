from dreal import Expression, Variable, tanh as dreal_tanh  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple

from cegus_lyapunov import PLyapunovLearner
from nnet_utils import DEVICE, LyapunovNet, NeuralNetRegressor


class LyapunovNetRegressor(NeuralNetRegressor, PLyapunovLearner):
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
        p_lya_p_hidden = (1.0 - lya_values**2) @ self._model.layer2.weight
        p_hidden_p_layer1 = (1.0 - torch.tanh(
            x_values @ self._model.layer1.weight.t() + self._model.layer1.bias)**2)
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

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expression:
        # Get weights and biases
        w1 = self.model.layer1.weight.data.cpu().numpy().squeeze()
        w2 = self.model.layer2.weight.data.cpu().numpy().squeeze()
        b1 = self.model.layer1.bias.data.cpu().numpy().squeeze()
        b2 = self.model.layer2.bias.data.cpu().numpy().squeeze()

        z1 = b1 + x_vars @ w1.T
        a1 = [dreal_tanh(z1[j]) for j in range(len(z1))]
        z2 = b2 + a1 @ w2.T
        lya_expr = dreal_tanh(z2)
        return lya_expr
    
    def lya_values(self, x_values: torch.Tensor) -> torch.Tensor:
        return self._model(x_values)

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expression]:
        raise NotImplementedError()

    def ctrl_values(self, x_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class KnownControl:
    """ Parameterized Lyapunov controller from the NeuRIPS 2022 paper """

    def __init__(self, C: torch.Tensor, K: torch.Tensor, b: torch.Tensor):
        self.C = torch.atleast_2d(C)
        self.K = torch.atleast_2d(K)
        self.b = b

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        linear = X @ self.K.T + self.b
        hidden = torch.atleast_2d(torch.tanh(linear))
        return hidden @ self.C.T

    def dreal_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expression]:
        C = self.C.cpu().numpy().squeeze()
        K = self.K.cpu().numpy().squeeze()
        b = self.b.cpu().numpy().squeeze()

        linear = np.atleast_1d(x_vars @ K + b)
        hidden = [dreal_tanh(v) for v in linear]
        ctrl_expr_arr = C.dot(hidden)
        return ctrl_expr_arr.tolist()


class KnownLyapunovNet(PLyapunovLearner):
    """ Parameterized Lyapunov function from the NeuRIPS 2022 paper """

    def __init__(self, W1, b1, W2, b2, ctrl: Optional[KnownControl] = None):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.ctrl = ctrl

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

    def fit_loop(self, X: torch.Tensor, y: torch.Tensor, max_epochs: int = 10, copy: bool = True) -> Sequence[float]:
        lya_risk = self._loss(X, y)
        return [lya_risk]

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        linear1 = X @ self.W1.T + self.b1
        hidden1 = torch.tanh(linear1)
        linear2 = hidden1 @ self.W2.T + self.b2
        lya = torch.tanh(linear2)
        return lya

    def lya_expr(self, x_vars: Sequence[Variable]) -> Expression:
        W1 = self.W1.cpu().numpy().squeeze()
        b1 = self.b1.cpu().numpy().squeeze()
        W2 = self.W2.cpu().numpy().squeeze()
        b2 = self.b2.cpu().numpy().squeeze()

        linear1 = x_vars @ W1.T + b1
        hidden1 = [dreal_tanh(expr) for expr in linear1]
        linear2 = hidden1 @ W2.T + b2
        lya_expr = dreal_tanh(linear2)
        return lya_expr
    
    def lya_values(self, x_values: torch.Tensor) -> torch.Tensor:
        return self.predict(x_values)

    def ctrl_exprs(self, x_vars: Sequence[Variable]) -> Sequence[Expression]:
        return [] if self.ctrl is None else self.ctrl.dreal_exprs(x_vars)
    
    def ctrl_values(self, x_values: torch.Tensor) -> torch.Tensor:
        return torch.tensor([], device=DEVICE).reshape(len(x_values), 0) if self.ctrl is None \
            else self.ctrl.apply(x_values)
