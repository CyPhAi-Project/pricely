from datetime import date
from pathlib import Path

import examples.neurips2022_inverted_pendulum as mod

from scripts import run_cegus, plot_phaseportrait_2d, validate_lip_bbox

OUT_DIR = Path(f"out/{date.today()}/{mod.__name__.split('.')[-1]}")
OUT_DIR.mkdir(exist_ok=True)

# Validate the provided Lipschitz constant(s) by evenly-spaced sampling
MAX_SAMPLES = 10**6
validate_lip_bbox.main(mod, MAX_SAMPLES)

# Synthesize a Lyapunov function
cand = run_cegus.main(mod, out_dir=OUT_DIR, max_num_samples=MAX_SAMPLES)

if mod.X_DIM == 2:
    plot_phaseportrait_2d.main(mod, cand, OUT_DIR)
