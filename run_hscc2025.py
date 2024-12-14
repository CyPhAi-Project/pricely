import os

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

from examples import \
    fossil_nonpoly0, fossil_nonpoly1, fossil_nonpoly2, fossil_nonpoly3, \
    fossil_poly1, fossil_poly2, fossil_poly3, fossil_poly4, \
    neurips2022_van_der_pol, neurips2022_unicycle_following, neurips2022_inverted_pendulum, \
    path_following_stanley

from scripts import run_cegus

MAX_SAMPLES = 10**6

trans_cases = [
    neurips2022_van_der_pol, neurips2022_unicycle_following, neurips2022_inverted_pendulum,
    path_following_stanley]

polys_cases = [
    fossil_nonpoly0, fossil_nonpoly1, fossil_nonpoly2, fossil_nonpoly3,
    fossil_poly2, fossil_poly3, fossil_poly4,
    fossil_poly1  # NOTE This benchmark will take a very long time.
]

for mod in trans_cases:
    print("\n")
    print(f" Benchmark: {mod.__name__} ".center(run_cegus.NCOLS, "#"))
    # Synthesize a Lyapunov function
    cand = run_cegus.execute(mod, out_dir=None, max_num_samples=MAX_SAMPLES)
