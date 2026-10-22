import json, numpy as np
from multiprocessing import Pool
import sar_cluster as S
K, D, SIG, FE = 1.4e4, 12.0, 0.15, 0.724
def job(a):
    kind, phi, H = a; m, e = S.sar(kind, phi, D, K, H, 1e5, sig=SIG, seeds=16, cycles=12)
    return dict(kind=kind, phi=phi, H=H, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    HS = [4.5e3, 7e3, 10e3, 15e3, 24e3]
    jobs = [("dilute", None, H) for H in HS] + [("loose", p, H) for p in (0.1, 0.2, 0.3) for H in HS]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("../data/nt_field.json", "w"), indent=1); print("done")
