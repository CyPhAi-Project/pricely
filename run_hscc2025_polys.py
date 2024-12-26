import csv
import os

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

from examples import \
    fossil_nonpoly0, fossil_nonpoly1, fossil_nonpoly2, fossil_nonpoly3, \
    fossil_poly1, fossil_poly2, fossil_poly3, fossil_poly4

from scripts import run_cegus

MAX_SAMPLES = 10**6

polys_cases = [
    fossil_nonpoly0,
    fossil_nonpoly1,
    fossil_nonpoly2,
    fossil_nonpoly3,
    # fossil_poly1,  # NOTE This benchmark will take a very long time.
    fossil_poly2,
    fossil_poly3,
    fossil_poly4
]

res_list = []

for mod in polys_cases:
    name = mod.__name__.split('.')[-1]
    print("\n")
    print(f" Benchmark: {name} ".center(run_cegus.NCOLS, "#"))
    # Synthesize a Lyapunov function
    stats = run_cegus.execute(mod, out_dir=None, max_num_samples=MAX_SAMPLES)
    res_list.append((name, mod.X_NORM_UB, stats))

with open('out/hscc2025_polys.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "r", "CEGuS Status", "Time", "k", "#samples", "#regions"])
    for name, r, stats in res_list:
        writer.writerow([
            name, r, stats.cegus_status, stats.cegus_time_usage,
            stats.last_epoch + 1, stats.num_samples, stats.num_regions])
