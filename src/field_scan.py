"""SAR versus field amplitude around the safety limit for the clinical and robust particles (required-field analysis),
and the clinical particle at the tolerated 100 kHz field for intermediate packing fractions (progressive aggregation)."""
import json, itertools, time
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG = 4.77e8, 0.724, 0.15
PART = {"clinical": (12.0, 1.4e4), "robust": (14.0, 2.6e4)}
def job(a):
    name, f, x, ph, seeds = a; d, K = PART[name]; H = x * LIMIT / f
    m, e = S.sar("dilute" if ph is None else "loose", ph, d, K, H, f, sig=SIG, seeds=seeds, cycles=10)
    return dict(particle=name, f=f, x=x, H=H, phi=ph, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    t0 = time.time()
    jobs = [("clinical", 1e5, 1.0, ph, 16) for ph in (None, 0.05, 0.1, 0.15, 0.2)]
    jobs += [(n, f, x, ph, 12) for n, f, x, ph in itertools.product(PART, (5e4, 7e4, 1e5, 1.4e5, 2e5), (0.5, 0.75, 1.0, 1.5, 2.0), (None, 0.1, 0.2))]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("../data/field_scan.json", "w"), indent=1); print("done", round(time.time() - t0))
