import json, itertools, numpy as np, time
from multiprocessing import Pool
import sar_cluster as S
K, D = 2.2e4, 13.0
PHI = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
def job(a):
    tag, kind, phi, d, k, Ms, H, f = a
    m, e = S.sar(kind, phi, d, k, H, f, Ms=Ms, seeds=8)
    return dict(tag=tag, kind=kind, phi=phi, f=f, SAR=m, se=e)
if __name__ == "__main__":
    jobs = [("GA", "loose", p, D, K, S.MS, 24e3, f) for p in PHI for f in (1e5, 3e5)]
    jobs += [("GA", "dilute", None, D, K, S.MS, 24e3, f) for f in (1e5, 3e5)]
    jobs += [("DC", "dilute", None, 10.0, 2.6e4, 2.8e5, 20e3, 7e5)] + [("DC", "loose", p, 10.0, 2.6e4, 2.8e5, 20e3, 7e5) for p in PHI]
    t0 = time.time()
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("../data/predict_phi.json", "w"), indent=1)
    for r in res: print(r)
    print("done", round(time.time() - t0))
