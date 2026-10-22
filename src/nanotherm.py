"""Particle-level model for the clinical magnetic fluid (aminosilane magnetite, core 12-15 nm, 100 kHz)."""
import json, sys, numpy as np
from multiprocessing import Pool
import sar_cluster as S
FE = 0.724                                   # g Fe per g Fe3O4
D, SIG = 12.0, 0.15
def job(a):
    kind, phi, K, H, seeds = a
    m, e = S.sar(kind, phi, D, K, H, 1e5, sig=SIG, seeds=seeds, cycles=12)
    return dict(kind=kind, phi=phi, K=K, H=H, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "K":
        jobs = [("dilute", None, K, 10e3, 16) for K in (0.8e4, 1.0e4, 1.2e4, 1.4e4, 1.7e4, 2.0e4)]
    else:
        K = float(sys.argv[2])
        jobs = [("dilute", None, K, H, 24) for H in (4e3, 4.5e3, 5e3, 10e3)] + \
               [("loose", p, K, H, 24) for p in (0.1, 0.2, 0.3, 0.4, 0.5) for H in (4e3, 4.5e3, 5e3)]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"../data/nanotherm_{stage}.json", "w"), indent=1)
    for r in res: print(r)
    print("done")
