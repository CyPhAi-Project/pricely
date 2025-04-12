import numpy as np
import unittest

from pricely.approx.boxes import AxisAlignedBoxes
from pricely.cegus_lyapunov import ROI, NDArrayFloat
from pricely.utils import gen_equispace_regions


class TestApproxBoxes(unittest.TestCase):
    def test_approx(self):
        X_LIM = np.array([
            [-1, -2, -3],
            [+1, +2, +3]])
        U_ROI = np.array([
            [-0.5, -2.5],
            [+0.5, +2.5]])

        def f_bbox(x: NDArrayFloat, u: NDArrayFloat) -> NDArrayFloat:
            return x

        def lip_bbox(x_regions: NDArrayFloat, u_roi: NDArrayFloat) -> NDArrayFloat:
            return np.ones((len(x_regions)))

        x_regions =gen_equispace_regions([2, 3, 4], X_LIM)
        u_values = np.zeros((len(x_regions), 2))
        approx = AxisAlignedBoxes(
            ROI(x_lim=X_LIM, abs_x_lb=2**-5), U_ROI, x_regions, u_values, f_bbox, lip_bbox)
        # TODO assertions on approximation


if __name__ == '__main__':
    unittest.main()
