from dreal import CheckSatisfiability, Config, Expression as Expr, sqrt as Sqrt, Variable, logical_and, logical_or, logical_iff  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from typing import NamedTuple, Optional

from pricely.cegus_lyapunov import NDArrayFloat, PLocalApprox, PLyapunovCandidate, PLyapunovVerifier


def pretty_sub(i: int) -> str:
    prefix = "" if i < 10 else pretty_sub(i // 10)
    return prefix + chr(0x2080 + (i % 10))


class DRealVars(NamedTuple):
    x: Variable
    der_lya: Variable
    abs_ub: Variable
    dxdt_s: Variable


class DRealInputs(NamedTuple):
    u: Variable


class SMTVerifier(PLyapunovVerifier):
    def __init__(
            self,
            x_roi: NDArrayFloat,
            u_dim: int = 0,
            abs_x_lb: ArrayLike = 2**-6,
            config: Config = None) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert np.all(np.asfarray(abs_x_lb) > 0.0) and np.all(np.isfinite(abs_x_lb))

        self._x_roi = x_roi

        x_dim = x_roi.shape[1]
        self._lya_var = Variable("V")
        self._decay_var = Variable("λ")
        self._lya_level_var = Variable("c")
        self._err_bnd_fun = Variable("ε(x,u)")
        self._in_reg_bit = Variable("x∈Rj", Variable.Binary)
        self._all_vars = [
            DRealVars(
                x=Variable(f"x{pretty_sub(i)}"),
                der_lya=Variable(f"∂V/∂x{pretty_sub(i)}"),
                abs_ub=Variable(f"b{pretty_sub(i)}"),
                dxdt_s=Variable(f"ŷ{pretty_sub(i)}(x,u)")
            ) for i in range(x_dim)
        ]

        self._all_inputs = [
            DRealInputs(
                u=Variable(f"u{pretty_sub(i)}"),
            ) for i in range(u_dim)
        ]

        self._smt_tpls = self._init_lyapunov_template(abs_x_lb)

        self._lya_cand_expr = Expr(0.0)
        self._decay_expr = Expr(0.0)
        self._lya_cand_level_ub = np.inf
        self._abs_x_ub = np.inf
        self._der_lya_cand_exprs = [Expr(0.0)]*x_dim
        self._ctrl_exprs = [Expr(0.0)]*u_dim

        if config is not None:
            self._config = config
        else:
            self._config = Config()
            self._config.use_polytope_in_forall = True
            self._config.use_local_optimization = True
            self._config.precision = 1e-9

    @property
    def x_dim(self) -> int:
        return len(self._all_vars)
    
    @property
    def u_dim(self) -> int:
        return len(self._all_inputs)

    def set_lyapunov_candidate(
            self, lya: PLyapunovCandidate):
        x_vars = [xi.x for xi in self._all_vars]
        self._lya_cand_expr = lya.lya_expr(x_vars)
        self._decay_expr = Expr(lya.lya_decay_rate())
        self._lya_cand_level_ub, self._abs_x_ub = \
            lya.find_sublevel_set_and_box(self._x_roi)
        self._der_lya_cand_exprs = [
            self._lya_cand_expr.Differentiate(x) for x in x_vars]
        if self.u_dim > 0:
            self._ctrl_exprs = lya.ctrl_exprs(x_vars)

    def reset_lyapunov_candidate(self):
        self._lya_cand_expr = Expr(0.0)
        self._decay_expr = Expr(0.0)
        self._lya_cand_level_ub = np.inf
        self._abs_x_ub = np.inf
        self._der_lya_cand_exprs = [Expr(0.0)]*self.x_dim
        self._ctrl_exprs = [Expr(0.0)]*self.u_dim

    def find_cex(
        self,
        f_approx_j: PLocalApprox
    ) -> Optional[NDArrayFloat]:
        assert self._lya_cand_expr is not None \
            and all(e is not None for e in self._der_lya_cand_exprs)

        verif_conds = self._inst_verif_conds(f_approx_j)

        smt_query = logical_or(*verif_conds)
        result = CheckSatisfiability(smt_query, self._config)
        if not result:
            return None
        else:
            x_vars  = [var.x for var in self._all_vars]
            box_np = np.asfarray(
                [[result[var].lb(), result[var].ub()] for var in x_vars]).transpose()
            return box_np
            

    def _init_lyapunov_template(self, abs_x_lb: ArrayLike):
        x_vars = [var.x for var in self._all_vars]
        der_lya_vars = [var.der_lya for var in self._all_vars]

        if np.isscalar(abs_x_lb):
            abs_x_lb_conds = [abs(x) >= Expr(abs_x_lb) for x in x_vars]
        else:
            abs_x_lb = np.asfarray(abs_x_lb)
            assert len(abs_x_lb) == len(x_vars)
            abs_x_lb_conds = [abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb)]
        abs_x_ub_conds = [abs(xi.x) <= xi.abs_ub for xi in self._all_vars]
        sublevel_set_cond = (self._lya_var <= self._lya_level_var)

        in_omega_pred = logical_and(
            *abs_x_lb_conds,
            *abs_x_ub_conds,
            sublevel_set_cond)

        lie_der_lya_hat = sum(var.der_lya*var.dxdt_s for var in self._all_vars)

        # Cauchy-Schwarz inequality for L2-norm
        der_lya_l2 = Sqrt(sum(e*e for e in der_lya_vars))
        # |∂V/∂x|ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) + λV(x)
        lie_der_lya_ub = der_lya_l2*self._err_bnd_fun + lie_der_lya_hat + self._decay_var*self._lya_var

        # Validity Conds: forall x in Rj.
        #   V(x) > 0 /\ ∂V/∂x⋅ŷ(x,k(x)) < 0 /\ |∂V/∂x|ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) < -λV(x)
        # SMT Conds: exists x in Rj.
        #   V(x)<= 0 \/ ∂V/∂x⋅ŷ(x,k(x))>= 0 \/ |∂V/∂x|ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) + λV(x) >= 0
        return [
            logical_and(in_omega_pred, self._in_reg_bit == 1, cond)
            for cond in [self._lya_var <= 0, lie_der_lya_hat >= 0, lie_der_lya_ub >= 0]]
    
    def _inst_verif_conds(self, f_approx_j: PLocalApprox):
        x_vars = [xi.x for xi in self._all_vars]
        u_vars = [ui.u for ui in self._all_inputs]
        set_domain = logical_iff(self._in_reg_bit == 1, f_approx_j.in_domain_pred(x_vars))
        tmp_tpls = (logical_and(set_domain, query_tpl) for query_tpl in self._smt_tpls)

        sub_pairs = \
            [(self._in_reg_bit, 1),
             (self._lya_var, self._lya_cand_expr),
             (self._decay_var, self._decay_expr),
             (self._err_bnd_fun, f_approx_j.error_bound_expr(x_vars, u_vars)),
             (self._lya_level_var, Expr(self._lya_cand_level_ub))] + \
            [(xi.der_lya, ei) for xi, ei in zip(self._all_vars, self._der_lya_cand_exprs)] + \
            [(xi.dxdt_s, fi) for xi, fi in zip(self._all_vars, f_approx_j.func_exprs(x_vars, u_vars))] + \
            [(ui.u, ei) for ui, ei in zip(self._all_inputs, self._ctrl_exprs)]

        if np.isscalar(self._abs_x_ub):
            sub_pairs.extend((xi.abs_ub, Expr(self._abs_x_ub)) for xi in self._all_vars)
        else:
            abs_x_ub = np.asfarray(self._abs_x_ub)
            assert len(abs_x_ub) == len(self._all_vars)
            sub_pairs.extend((xi.abs_ub, Expr(ub)) for xi, ub in zip(self._all_vars, abs_x_ub))

        sub_dict = dict(sub_pairs)
        if hasattr(f_approx_j, "_sel_var"):
            verif_conds = (query_tpl.Substitute(sub_dict) for query_tpl in tmp_tpls)
            return [
                logical_and(*(vc.Substitute({f_approx_j._sel_var: Expr(k)}) for k in range(self.x_dim + 1)))  #type:ignore
                for vc in verif_conds]
        return [query_tpl.Substitute(sub_dict) for query_tpl in tmp_tpls]


def test_smt_verifier():
    X_ROI = np.asfarray([
        [-1, -2, -3],
        [+1, +2, +3]
    ])
    verifier = SMTVerifier(X_ROI, u_dim=2, abs_x_lb=2**-4)
    print(*verifier._smt_tpls, sep="\n")


def test_substitution():
    from pricely.learner_mock import MockQuadraticLearner
    from pricely.approx.boxes import ConstantApprox
    X_ROI = np.asfarray([
        [-1, -2, -3],
        [+1, +2, +3]
    ])
    X_DIM = X_ROI.shape[1]
    U_DIM = 2

    pd_mat = np.eye(X_DIM)
    ctrl_mat = np.eye(U_DIM, X_DIM)
    learner = MockQuadraticLearner(pd_mat=pd_mat, ctrl_mat=ctrl_mat)

    verifier = SMTVerifier(X_ROI, u_dim=U_DIM, abs_x_lb=2**-4)
    verifier.set_lyapunov_candidate(learner.get_candidate())

    region = np.row_stack((np.zeros(X_DIM), X_ROI)).reshape((3, X_DIM))
    approx = ConstantApprox(region, ctrl_mat @ region[0], region[0], 1.0)
    verif_conds = verifier._inst_verif_conds(approx)
    print(*verif_conds, sep='\n')


if __name__ == "__main__":
    test_smt_verifier()
    test_substitution()
