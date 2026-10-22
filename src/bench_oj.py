import json, numpy as np
from multiprocessing import Pool
import sar_cluster as S
def job(a):
    kind, phi, N, H = a
    m, e = S.sar(kind, phi, 22.0, 1.1e4, H, 3e5, N=N, sig=0.0, gap=2.0, seeds=10, Ms=4.8e5)
    return dict(kind=kind, phi=phi, N=N, H=H, SAR=m, se=e)
if __name__ == "__main__":
    jobs = [(k, p, n, H) for (k, p, n) in (("dilute", None, 50), ("chain", None, 20), ("loose", 0.40, 50)) for H in (12e3, 24e3)]
    with Pool(2) as p: res = p.map(job, jobs)
    json.dump(res, open("../data/bench_oj.json", "w"), indent=1)
    for r in res: print(r)
    print("done")
