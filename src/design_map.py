import json, itertools, numpy as np, time
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG = 4.77e8, 0.724, 0.15            # IEC 60601-2-33 first-level H*f (Pilpilidis 2025)
DS = [10.0, 12.0, 14.0, 16.0, 18.0]; KS = [0.8e4, 1.2e4, 1.6e4, 2.0e4, 2.6e4, 3.4e4]; FS = [5e4, 1e5, 2e5]; ST = [None, 0.1, 0.2]
def job(a):
    d, K, f, ph = a; H = LIMIT / f
    m, e = S.sar("dilute" if ph is None else "loose", ph, d, K, H, f, sig=SIG, seeds=8, cycles=10)
    return dict(d=d, K=K, f=f, H=H, phi=ph, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    t0 = time.time(); jobs = list(itertools.product(DS, KS, FS, ST))
    with Pool(2) as p: res = p.map(job, jobs, chunksize=2)
    json.dump(res, open("../data/design_map.json", "w"), indent=1); print("done", round(time.time() - t0))
