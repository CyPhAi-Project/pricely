from dreal import Variable  # type: ignore
import numpy as np
from scipy.spatial import Delaunay
import unittest

from pricely.approx.boxes import ConstantApprox
from pricely.approx.simplices import AnySimplexConstant
from pricely.cegus_lyapunov import ROI 
from pricely.learner.mock import MockQuadraticLearner
from pricely.verifier.smt_dreal import SMTVerifier
from pricely.utils import pretty_sub

X_LIM = np.array([
    [-1, -1.5, -3],
    [+1, +1.5, +3]
])
X_DIM = X_LIM.shape[1]


class TestSMTVerifier(unittest.TestCase):
    def test_substitution(self):
        U_DIM = 2
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(X_DIM)]

        pd_mat = np.eye(X_DIM)
        ctrl_mat = -np.eye(U_DIM, X_DIM)
        learner = MockQuadraticLearner(pd_mat=pd_mat, ctrl_mat=ctrl_mat)

        x_roi = ROI(x_lim=X_LIM, abs_x_lb=2**-4)
        verifier = SMTVerifier(x_roi, u_dim=U_DIM)
        verifier.set_lyapunov_candidate(learner.get_candidate())

        region = np.vstack((np.zeros(X_DIM), X_LIM)).reshape((3, X_DIM))
        approx = ConstantApprox(region, ctrl_mat @ region[0], region[0], 1.0)
        verif_conds = verifier._inst_verif_conds(approx, x_vars)
        # TODO assertions on the generated SMT queries

    def test_infeasible(self):
        u_dim = 0
        pd_mat = np.eye(X_DIM)
        learner = MockQuadraticLearner(pd_mat=pd_mat)
        x_roi = ROI(x_lim=X_LIM, abs_x_lb=2**-4)
        verifier = SMTVerifier(x_roi)
        verifier.set_lyapunov_candidate(learner.get_candidate())

        x_values = np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 0]])
        y_values = -1*np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 0]])
        
        triangulation = Delaunay(points=x_values, qhull_options="Q12 QJ")
        trans = triangulation.transform[1]

        lips = 1.125
        approx = AnySimplexConstant(
            trans_barycentric=trans,
            x_values=x_values,
            u_values=np.zeros(shape=(len(x_values), u_dim)),
            y_values=y_values,
            hash_key=1,
            lip=lips)

        res = verifier.find_cex(approx)
        self.assertIsNone(res)


if __name__ == '__main__':
    unittest.main()
