"""Time-step convergence with more aggregates: 300 vs 600 steps per cycle (clinical particle, 100 kHz, 4.77 kA/m)."""
import json
from multiprocessing import Pool
import sar_cluster as S
def job(a):
    ph, st = a; m, e = S.sar("loose", ph, 12.0, 1.4e4, 4770.0, 1e5, sig=0.15, seeds=24, cycles=10, steps=st)
    return dict(phi=ph, steps=st, SAR_Fe=m / 0.724, se_Fe=e / 0.724)
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(job, [(0.1, 300), (0.1, 600), (0.2, 300), (0.2, 600)], chunksize=1)
    json.dump(res, open("../data/conv2.json", "w"), indent=1); print(res, "done")
