from dreal import CheckSatisfiability, Config, Expression as Expr, sqrt as Sqrt, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from typing import NamedTuple, Optional, Sequence, Union

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


class ConfigTuple(NamedTuple):
    precision: float = 1e-9
    use_polytope: bool = True
    use_polytope_in_forall: bool = False
    use_worklist_fixpoint: bool = False
    use_local_optimization: bool = True
    number_of_jobs: int = 1
    nlopt_ftol_rel: float = 1e-06
    nlopt_ftol_abs: float = 1e-06
    nlopt_maxeval: int = 100
    nlopt_maxtime: float = 0.01


def from_dreal(config: Config) -> ConfigTuple:
    return ConfigTuple(
        precision=config.precision,
        use_polytope=config.use_polytope,
        use_polytope_in_forall=config.use_polytope_in_forall,
        use_worklist_fixpoint=config.use_worklist_fixpoint,
        use_local_optimization=config.use_local_optimization,
        number_of_jobs=config.number_of_jobs,
        nlopt_ftol_rel=config.nlopt_ftol_rel,
        nlopt_ftol_abs=config.nlopt_ftol_abs,
        nlopt_maxeval=config.nlopt_maxeval,
        nlopt_maxtime=config.nlopt_maxtime)


def to_dreal(conf_tup: ConfigTuple) -> Config:
    config = Config()
    config.precision=conf_tup.precision
    config.use_polytope=conf_tup.use_polytope
    config.use_polytope_in_forall=conf_tup.use_polytope_in_forall
    config.use_worklist_fixpoint=conf_tup.use_worklist_fixpoint
    config.use_local_optimization=conf_tup.use_local_optimization
    config.number_of_jobs=conf_tup.number_of_jobs
    config.nlopt_ftol_rel=conf_tup.nlopt_ftol_rel
    config.nlopt_ftol_abs=conf_tup.nlopt_ftol_abs
    config.nlopt_maxeval=conf_tup.nlopt_maxeval
    config.nlopt_maxtime=conf_tup.nlopt_maxtime
    return config


class SMTVerifier(PLyapunovVerifier):
    def __init__(
            self,
            x_roi: NDArrayFloat,
            u_dim: int = 0,
            abs_x_lb: ArrayLike = 2**-6,
            config: Union[Config, ConfigTuple] = ConfigTuple()) -> None:
        assert x_roi.shape[0] == 2 and x_roi.shape[1] >= 1
        assert np.all(np.asfarray(abs_x_lb) > 0.0) and np.all(np.isfinite(abs_x_lb))

        self._x_roi = x_roi
        self._u_dim = u_dim
        self._abs_x_lb = abs_x_lb
        if isinstance(config, ConfigTuple):
            self._conf_tup = config
        else:  # Explicit copy and convert to Python tuple for pickling
            self._conf_tup = from_dreal(config)
        self._lya_cand = None

    @property
    def x_dim(self) -> int:
        return self._x_roi.shape[1]
    
    @property
    def u_dim(self) -> int:
        return self._u_dim

    def set_lyapunov_candidate(
            self, lya: PLyapunovCandidate):
        self._lya_cand = lya

    def find_cex(
        self,
        f_approx_j: PLocalApprox
    ) -> Optional[NDArrayFloat]:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(self.x_dim)]
        smt_query = self._inst_verif_conds(f_approx_j, x_vars)

        result = CheckSatisfiability(smt_query, to_dreal(self._conf_tup))
        if not result:
            return None
        else:
            box_np = np.asfarray(
                [[result[var].lb(), result[var].ub()] for var in x_vars]).transpose()
            return box_np
        
    def _inst_verif_conds(
            self, f_approx_j: PLocalApprox,
            x_vars: Sequence[Variable]):
        assert self._lya_cand is not None
        lya =  self._lya_cand

        # Add our Lyapunov criteria
        u_vars = [Variable(f"u{pretty_sub(i)}") for i in range(self.u_dim)]  # Temp variables for input
        y_vars = [Variable(f"ŷ{pretty_sub(i)}") for i in range(self.x_dim)]
        err_bnd_var = Variable("ε(x,k(x))")
        lya_cand_expr = lya.lya_expr(x_vars)
        der_lya_cand_exprs = [lya_cand_expr.Differentiate(x) for x in x_vars]
        decay_expr = Expr(lya.lya_decay_rate())
        ## Build ∂V/∂x⋅ŷ(x,k(x))
        lie_der_lya_hat: Expr = sum(
            der_lya_i*y_i
            for der_lya_i, y_i in zip(der_lya_cand_exprs, y_vars))
        ## Build ||∂V/∂x||
        der_lya_l2 = Sqrt(sum(e*e for e in der_lya_cand_exprs))
        ## Build ||∂V/∂x||ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) + λV(x)
        lie_der_lya_ub = der_lya_l2*err_bnd_var + lie_der_lya_hat + decay_expr*lya_cand_expr

        # Validity Conds: forall x in Rj.
        #   V(x) > 0 /\ ∂V/∂x⋅ŷ(x,k(x)) < 0 /\ |∂V/∂x|ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) < -λV(x)
        # SMT Conds: exists x in Rj.
        #   V(x)<= 0 \/ ∂V/∂x⋅ŷ(x,k(x))>= 0 \/ |∂V/∂x|ε(x,k(x)) + ∂V/∂x⋅ŷ(x,k(x)) + λV(x) >= 0
        falsify_lya_pos = (lya_cand_expr <= 0)
        falsify_der_lya_hat_tpl = (lie_der_lya_hat >= 0)
        falsify_bbox_cond_tpl = (lie_der_lya_ub >= 0)

        # Must falsify all approximations
        vcs_der_lya_hat = []
        vcs_bbox_cond = []
        for k in range(f_approx_j.num_approxes):
            y_exprs = f_approx_j.func_exprs(x_vars, u_vars, k)
            err_bnd_expr = f_approx_j.error_bound_expr(x_vars, u_vars, k)
            sub_approx = dict(
                [(yi, ei) for yi, ei in zip(y_vars, y_exprs)] +
                [(err_bnd_var, err_bnd_expr)])
            vcs_der_lya_hat.append(falsify_der_lya_hat_tpl.Substitute(sub_approx))  #type:ignore
            vcs_bbox_cond.append(falsify_bbox_cond_tpl.Substitute(sub_approx))  #type:ignore
        falsify_der_lya_hat = logical_and(*vcs_der_lya_hat)
        falsify_bbox_cond = logical_and(*vcs_bbox_cond)

        smt_query_tpl = logical_or(
            falsify_lya_pos, falsify_der_lya_hat, falsify_bbox_cond)

        # Substitute with the current controller
        ctrl_exprs = lya.ctrl_exprs(x_vars)
        sub_input = dict((ui, ei) for ui, ei in zip(u_vars, ctrl_exprs))
        smt_query = smt_query_tpl.Substitute(sub_input)

        # Add constraints of the region
        region_pred_list = []
        ## Add the region of interest X
        if np.isscalar(self._abs_x_lb):
            abs_x_lb_conds = (abs(x) >= Expr(self._abs_x_lb) for x in x_vars)
        else:
            abs_x_lb = np.asfarray(self._abs_x_lb)
            assert len(abs_x_lb) == len(x_vars)
            abs_x_lb_conds = (abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb))
        region_pred_list.extend(abs_x_lb_conds)

        lya_cand_level_ub, abs_x_ub = \
            lya.find_sublevel_set_and_box(self._x_roi)
        if np.isscalar(abs_x_ub):
            abs_x_ub_conds = (abs(x) <= Expr(abs_x_ub) for x in x_vars)
        else:
            abs_x_ub = np.asfarray(abs_x_ub)
            assert len(abs_x_ub) == len(x_vars)
            abs_x_ub_conds = (abs(x) <= Expr(ub) for x, ub in zip(x_vars, abs_x_ub))
        region_pred_list.extend(abs_x_ub_conds)

        sublevel_set_cond = (lya.lya_expr(x_vars) <= Expr(lya_cand_level_ub))
        region_pred_list.append(sublevel_set_cond)

        ## Add the region Rj
        region_pred_list.append(f_approx_j.in_domain_pred(x_vars))
        return logical_and(*region_pred_list, smt_query)


def test_substitution():
    from pricely.learner_mock import MockQuadraticLearner
    from pricely.approx.boxes import ConstantApprox
    X_ROI = np.asfarray([
        [-1, -1.5, -3],
        [+1, +1.5, +3]
    ])
    X_DIM = X_ROI.shape[1]
    U_DIM = 2
    x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(X_DIM)]

    pd_mat = np.eye(X_DIM)
    ctrl_mat = -np.eye(U_DIM, X_DIM)
    learner = MockQuadraticLearner(pd_mat=pd_mat, ctrl_mat=ctrl_mat)

    verifier = SMTVerifier(X_ROI, u_dim=U_DIM, abs_x_lb=2**-4)
    verifier.set_lyapunov_candidate(learner.get_candidate())

    region = np.row_stack((np.zeros(X_DIM), X_ROI)).reshape((3, X_DIM))
    approx = ConstantApprox(region, ctrl_mat @ region[0], region[0], 1.0)
    verif_conds = verifier._inst_verif_conds(approx, x_vars)
    print(verif_conds, sep='\n')


if __name__ == "__main__":
    test_substitution()
