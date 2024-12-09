from dreal import Box, CheckSatisfiability, Config, Expression as Expr, Variable, logical_and, logical_or  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Sequence, Tuple

from pricely.cegus_lyapunov import NDArrayIndex, NDArrayFloat, ROI


def _gen_neg_lya_cond(
        x_vars: Sequence[Variable],
        dxdt_exprs: Sequence[Expr],
        lya_expr: Expr,
        lya_decay_rate: float) -> Expr:
    assert lya_decay_rate >= 0.0
    der_lya = [lya_expr.Differentiate(x) for x in x_vars]
    lie_der_lya = sum(
        der_lya_i*dxdt_i
        for der_lya_i, dxdt_i in zip(der_lya, dxdt_exprs))
    return logical_or(lya_expr <= 0.0, lie_der_lya + lya_decay_rate*lya_expr >= 0.0)


def pretty_sub(i: int) -> str:
    """ Convert integers to subscripted numbers """
    assert i >= 0
    rev_str = [chr(0x2080 + (i % 10))]
    while i >= 10:
        i = i // 10
        rev_str.append(chr(0x2080 + (i % 10)))
    return "".join(reversed(rev_str))


def exclude_rows(
        x_values: NDArrayFloat, u_values: NDArrayFloat, y_values: NDArrayFloat,
        exclude_indices: NDArrayIndex) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    row_mask = np.ones(shape=len(x_values), dtype=bool)
    row_mask[exclude_indices] = False
    return x_values[row_mask], u_values[row_mask], y_values[row_mask]


def check_lyapunov_roi(
    x_vars: Sequence[Variable],
    dxdt_exprs: Sequence[Expr],
    lya_expr: Expr,
    x_roi: ROI,
    lya_decay_rate: float = 0.0,
    config: Config = Config()
) -> Optional[Box]:
    x_lim, abs_x_lb, (x_norm_lb, x_norm_ub) = x_roi
    assert x_lim.shape == (2, len(x_vars))
    assert 0.0 <= x_norm_lb <= x_norm_ub

    lb_conds = [x >= Expr(lb) for x, lb in zip(x_vars, x_lim[0])]
    ub_conds = [x <= Expr(ub) for x, ub in zip(x_vars, x_lim[1])]

    abs_lb_conds = []
    if np.isscalar(abs_x_lb):
        abs_lb_conds.extend(abs(x) >= Expr(abs_x_lb) for x in x_vars)
    else:
        abs_x_lb = np.asfarray(abs_x_lb)
        assert len(abs_x_lb) == len(x_vars)
        abs_lb_conds.extend(
            abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb))
    exclude_rect = logical_or(*abs_lb_conds)
    if x_norm_lb > 0.0 or np.isfinite(x_norm_ub):
        norm_sq = sum(x**2 for x in x_vars)
        norm_conds = [norm_sq >= x_norm_lb**2, norm_sq <= x_norm_ub**2]
    else:
        norm_conds = []

    in_roi_pred = logical_and(*lb_conds, *ub_conds, exclude_rect, *norm_conds)

    neg_lya_cond = _gen_neg_lya_cond(x_vars, dxdt_exprs, lya_expr, lya_decay_rate)
    smt_query = logical_and(in_roi_pred, neg_lya_cond)
    result = CheckSatisfiability(smt_query, config)
    return result


def check_lyapunov_sublevel_set(
    x_vars: Sequence[Variable],
    dxdt_exprs: Sequence[Expr],
    lya_expr: Expr,
    lya_decay_rate: float = 0.0,
    level_ub: float = np.inf,
    abs_x_lb: ArrayLike = 2**-6,
    abs_x_ub: ArrayLike = 2**6,
    config: Config = Config()
) -> Optional[Box]:
    assert np.all(0.0 < np.asfarray(abs_x_lb))
    assert np.all(np.isfinite(abs_x_ub))
    assert np.all(np.asfarray(abs_x_lb) < np.asfarray(abs_x_ub))

    sublevel_set_cond = lya_expr <= Expr(level_ub)

    # Add the range on the absolute values of x to limit the search space.
    abs_range_conds = []
    if np.isscalar(abs_x_lb):
        abs_range_conds.extend(abs(x) >= Expr(abs_x_lb) for x in x_vars)
    else:
        abs_x_lb = np.asfarray(abs_x_lb)
        assert len(abs_x_lb) == len(x_vars)
        abs_range_conds.extend(
            abs(x) >= Expr(lb) for x, lb in zip(x_vars, abs_x_lb))

    raise NotImplementedError("Computing a bounding box of the sublevel set is not supported.")
    if np.isscalar(abs_x_ub):
        abs_range_conds.extend(abs(x) <= Expr(abs_x_ub) for x in x_vars)
    else:
        abs_x_ub = np.asfarray(abs_x_ub)
        assert len(abs_x_ub) == len(x_vars)
        abs_range_conds.extend(
            abs(x) <= Expr(ub) for x, ub in zip(x_vars, abs_x_ub))

    in_omega_pred = logical_and(
        *abs_range_conds,
        sublevel_set_cond)

    neg_lya_cond = _gen_neg_lya_cond(x_vars, dxdt_exprs, lya_expr, lya_decay_rate)
    smt_query = logical_and(in_omega_pred, neg_lya_cond)
    result = CheckSatisfiability(smt_query, config)
    return result


def cartesian_prod(*arrays) -> np.ndarray:
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m, k+1:] = out[0:m, k+1:]
    return out


def gen_equispace_regions(
        x_part: Sequence[int],
        x_lim: np.ndarray) -> np.ndarray:
    """ Generate reference values and bounds of x inside ROI """
    assert x_lim.shape == (2, len(x_part))
    x_dim = len(x_part)
    axes_cuts = (np.linspace(
        x_lim[0, i], x_lim[1, i], x_part[i]+1) for i in range(x_dim))
    bound_pts = cartesian_prod(
        *axes_cuts).reshape(tuple(n+1 for n in x_part) + (x_dim,))
    lb_pts = bound_pts[(slice(0, -1),)*x_dim].reshape((-1, x_dim))
    ub_pts = bound_pts[(slice(1, None),)*x_dim].reshape((-1, x_dim))
    x = (lb_pts + ub_pts) / 2
    return np.stack((x, lb_pts, ub_pts), axis=1)
