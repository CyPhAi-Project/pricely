import csv
import os

# Disable all tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

from examples import \
    neurips2022_van_der_pol, neurips2022_unicycle_following, neurips2022_inverted_pendulum, \
    path_following_stanley

from scripts import run_cegus

MAX_SAMPLES = 5*10**5

trans_cases = [
    neurips2022_van_der_pol, neurips2022_unicycle_following, neurips2022_inverted_pendulum,
    path_following_stanley]

res_list = []

for mod in trans_cases:
    name = mod.__name__.split('.')[-1]
    print("\n")
    print(f" Benchmark: {name} ".center(run_cegus.NCOLS, "#"))
    # Synthesize a Lyapunov function
    stats = run_cegus.execute(mod, out_dir=None, max_num_samples=MAX_SAMPLES)
    res_list.append((name, stats))

with open('out/hscc2025_trans.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "CEGuS Status", "Time", "k", "|S_L|", "|S|", "|C|"])
    for name, stats in res_list:
        writer.writerow([
            name, stats.cegus_status, stats.cegus_time_usage,
            stats.last_epoch + 1, stats.num_samples_learn, stats.num_samples, stats.num_regions])
