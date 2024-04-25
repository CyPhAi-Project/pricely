import gurobipy as gp
import numpy as np
from tqdm import tqdm

from pricely.approx.simplices import SimplicialComplex
from pricely.candidates import QuadraticLyapunov


BARY = {"lb": 0.0, "ub": 1.0, "vtype": gp.GRB.CONTINUOUS}
FREE = {"lb": -np.inf, "ub": np.inf, "vtype": gp.GRB.CONTINUOUS}


def init_approx_dynamics(approx: SimplicialComplex, max_epoch: int=100, n_jobs: int=1):
    cand = QuadraticLyapunov(np.eye(approx.x_dim))

    m = gp.Model("Simplex Covering", env=gp.Env())
    bary_var = m.addMVar(name="λ", shape=(approx.x_dim+1,), **BARY)
    w_var = m.addVar(name=f"w", **FREE)
    m.addConstr(np.ones(bary_var.shape) @ bary_var == 1.0)

    for epoch in tqdm(iter(range(1, max_epoch+1)), desc= "Presolve", ascii=True):
        cex_list = []
        prev_cons = []
        for i in tqdm(range(len(approx)), desc= "Simplex Covering", ascii=True, leave=False):
            local_approx = approx[i]
            m.remove(prev_cons)

            x_vertices = local_approx._x_values.transpose()
            radii = np.linalg.norm(local_approx._y_values, axis=1) / local_approx._lip

            theta = x_vertices @ bary_var
            rhs_arr = np.sum(x_vertices**2, axis=0) - radii**2
            lhs_arr = 2*(x_vertices.T @ theta) + w_var  # type: ignore
            cons = m.addConstr(lhs_arr <= rhs_arr)
            prev_cons = cons.tolist()

            m.setObjective(
                expr=theta @ theta + w_var,
                sense=gp.GRB.MAXIMIZE)
            m.optimize()
            if m.status != gp.GRB.OPTIMAL:
                raise RuntimeError('Gurobi ended with status %d.' % m.status)
            if m.objVal <= 0.0:
                continue
            # else:
            x_cex = x_vertices @ bary_var.X
            cex_list.append((i, np.row_stack((x_cex, x_cex))))

        if not cex_list:
            break
        # else:
        approx.add(cex_list, cand)
