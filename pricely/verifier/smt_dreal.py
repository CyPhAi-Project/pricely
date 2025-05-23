from dreal import CheckSatisfiability, Config, Expression as Expr, sqrt as Sqrt, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from typing import NamedTuple, Optional, Sequence, Union

from pricely.cegus_lyapunov import NDArrayFloat, ROI, PLocalApprox, PLyapunovCandidate, PLyapunovVerifier
from pricely.utils import pretty_sub


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
            x_roi: ROI,
            u_dim: int = 0,
            config: Union[Config, ConfigTuple] = ConfigTuple()) -> None:
        x_lim, abs_x_lb, (x_norm_lb, x_norm_ub) = x_roi
        assert x_lim.shape[0] == 2 and x_lim.shape[1] >= 1
        assert np.all(np.array(abs_x_lb) >= 0.0) and np.all(np.isfinite(abs_x_lb))
        assert 0.0 <= x_norm_lb <= x_norm_ub

        self._x_roi = x_roi
        self._u_dim = u_dim
        self._abs_x_lb = abs_x_lb
        self._x_norm_lb = x_norm_lb
        self._x_norm_ub = x_norm_ub
        if isinstance(config, ConfigTuple):
            self._conf_tup = config
        else:  # Explicit copy and convert to Python tuple for pickling
            self._conf_tup = from_dreal(config)
        self._lya_cand: Optional[PLyapunovCandidate] = None

    @property
    def x_dim(self) -> int:
        return self._x_roi.x_lim.shape[1]

    @property
    def u_dim(self) -> int:
        return self._u_dim

    def set_lyapunov_candidate(
            self, cand: PLyapunovCandidate):
        self._lya_cand = cand

    def find_cex(
        self,
        approx_j: PLocalApprox
    ) -> Optional[NDArrayFloat]:
        x_vars = [Variable(f"x{pretty_sub(i)}") for i in range(self.x_dim)]
        smt_query = self._inst_verif_conds(approx_j, x_vars)

        result = CheckSatisfiability(smt_query, to_dreal(self._conf_tup))
        if not result:
            return None
        else:
            box_np = np.array(
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
        lya_cand_expr: Expr = lya.lya_expr(x_vars)
        der_lya_cand_exprs = [lya_cand_expr.Differentiate(x) for x in x_vars]
        decay_expr = Expr(lya.lya_decay_rate())
        ## Build ∇V(x)⋅ŷ
        lie_der_lya_hat: Expr = sum(
            der_lya_i*y_i
            for der_lya_i, y_i in zip(der_lya_cand_exprs, y_vars))
        ## Build ‖∇V(x)‖
        der_lya_l2 = Sqrt(sum(e*e for e in der_lya_cand_exprs))
        ## Build ‖∇V(x)‖ε(x,k(x)) + ∇V(x)⋅ŷ + λV(x)
        lie_der_lya_ub = der_lya_l2*err_bnd_var + lie_der_lya_hat + decay_expr*lya_cand_expr

        # Validity Conds: forall x in Rj.
        #   V(x) > 0 /\ ∇V(x)⋅ŷ < 0 /\ ‖∇V(x)‖ε(x,k(x)) + ∇V(x)⋅ŷ < -λV(x)
        # SMT Conds: exists x in Rj.
        #   V(x)<= 0 \/ ∇V(x)⋅ŷ>= 0 \/ ‖∇V(x)‖ε(x,k(x)) + ∇V(x)⋅ŷ + λV(x) >= 0)
        falsify_lya_pos = (lya_cand_expr <= 0)
        falsify_der_lya_hat_tpl = (lie_der_lya_hat >= 0)
        falsify_bbox_cond_tpl = (lie_der_lya_ub >= 0)

        # Must falsify all approximations
        vcs_der_lya_hat = []
        vcs_bbox_cond = []
        for k in range(f_approx_j.num_approxes):
            y_exprs: Sequence[Expr] = f_approx_j.func_exprs(x_vars, u_vars, k)
            err_bnd_expr: Expr = f_approx_j.error_bound_expr(x_vars, u_vars, k)
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
        ctrl_exprs: Sequence[Expr] = lya.ctrl_exprs(x_vars)
        sub_input = dict((ui, ei) for ui, ei in zip(u_vars, ctrl_exprs))
        smt_query = smt_query_tpl.Substitute(sub_input)

        # Add constraints of the region
        region_pred_list = []
        # Add the region of interest X
        if np.isscalar(self._abs_x_lb):
            abs_x_lb_conds = (abs(x) >= Expr(self._abs_x_lb) for x in x_vars)
        else:
            abs_x_lb = np.array(self._abs_x_lb)
            assert len(abs_x_lb) == len(x_vars)
            abs_x_lb_conds = (abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb))
        exclude_rect = logical_or(*abs_x_lb_conds)
        region_pred_list.append(exclude_rect)

        x_lim_lb_conds = (x >= Expr(lb) for x, lb in zip(x_vars, self._x_roi.x_lim[0]))
        region_pred_list.extend(x_lim_lb_conds)
        x_lim_ub_conds = (x <= Expr(ub) for x, ub in zip(x_vars, self._x_roi.x_lim[1]))
        region_pred_list.extend(x_lim_ub_conds)

        if self._x_norm_lb > 0.0 or np.isfinite(self._x_norm_ub):
            norm_sq = sum(x**2 for x in x_vars)
            x_norm_lb_cond = norm_sq >= self._x_norm_lb**2
            x_norm_ub_cond = norm_sq <= self._x_norm_ub**2
            region_pred_list.append(x_norm_lb_cond)
            region_pred_list.append(x_norm_ub_cond)

        ## Add the region Rj
        region_pred_list.append(f_approx_j.in_domain_pred(x_vars))
        return logical_and(*region_pred_list, smt_query)
