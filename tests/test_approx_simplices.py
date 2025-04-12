from dreal import Variable  # type: ignore
import numpy as np
import unittest

from pricely.approx.simplices import SimplicialComplex
from pricely.cegus_lyapunov import ROI, NDArrayFloat
from pricely.utils import cartesian_prod


class TestApproxSimplices(unittest.TestCase):
    def test_approx(self):
        X_LIM = np.array([
            [-1, -2],
            [+1, +2]])
        U_ROI = np.array([
            [-0.5],
            [+0.5]])
        X_DIM = X_LIM.shape[1]
        X_ROI = ROI(x_lim=X_LIM, x_norm_lim=(0.125, 1.5), abs_x_lb=0.0)

        def f_bbox(x: NDArrayFloat, u: NDArrayFloat) -> NDArrayFloat:
            dxdt = np.empty_like(x)
            dxdt[:, 0] = -x[:, 0]
            dxdt[:, 1] = -0.5*x[:, 1]**3
            return dxdt

        def lip_bbox(x_regions: NDArrayFloat, u_roi: NDArrayFloat) -> NDArrayFloat:
            max_abs_x: NDArrayFloat = abs(x_regions).max(axis=1)
            max_abs_x[:, 1] = 1.5 * max_abs_x[:, 1]**2
            return max_abs_x.max(axis=1)

        axis_cuts = np.linspace(start=X_LIM[0], stop=X_LIM[1], num=5).transpose()
        x_values = cartesian_prod(*axis_cuts)
        u_values = np.zeros((len(x_values), U_ROI.shape[1]))
        approx = SimplicialComplex(
            X_ROI, U_ROI, x_values, u_values, f_bbox, lip_bbox)

        item = 3
        local_approx = approx[item]
        x_vars = [Variable(f"x{i}") for i in range(X_LIM.shape[1])]
        u_vars = [Variable(f"u{i}") for i in range(U_ROI.shape[1])]

        # TODO add assertions
        print(local_approx._x_values)
        print(local_approx._y_values)
        print(local_approx.in_domain_pred(x_vars))
        print(local_approx.func_exprs(x_vars, u_vars, 1))
        print(local_approx.error_bound_expr(x_vars, u_vars, 1))


if __name__ == '__main__':
    unittest.main()
