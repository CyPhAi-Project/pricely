from datetime import date
from pathlib import Path

import examples.neurips2022_van_der_pol as mod

from scripts import run_cegus, plot_phaseportrait_2d, validate_lip_bbox, validate_lya_cand

OUT_DIR = Path(f"out/{date.today()}/{mod.__name__.split('.')[-1]}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 1e-4
MAX_SAMPLES = 5*10**5

# Validate the provided Lipschitz constant(s) by evenly-spaced sampling
validate_lip_bbox.execute(mod, MAX_SAMPLES)

# Synthesize a Lyapunov function
stats = run_cegus.execute(mod, out_dir=OUT_DIR, delta=DELTA, max_num_samples=MAX_SAMPLES)
cand = stats.last_candidate

# Validate the synthesized Lyapunov function if the white-box model (RHS of ODE) is given.
if cand is not None and hasattr(mod, "f_expr"):
    validate_lya_cand.execute(mod, cand)

if mod.X_DIM == 2:
    plot_phaseportrait_2d.execute(mod, cand, OUT_DIR)

print(" Final Statistics ".center(run_cegus.NCOLS, "#"))
print(f"CEGuS Status: {stats.cegus_status}. "
    f"#iterations: {stats.last_epoch + 1}. "
    f"#samples for learner: {stats.num_samples_learn}. "
    f"#samples: {stats.num_samples}. "
    f"#regions: {stats.num_regions}. "
    f"Time: {stats.cegus_time_usage:.3f}s")
