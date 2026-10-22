import json, numpy as np
from multiprocessing import Pool
import sar_cluster as S
def job(a):
    kind, phi = a; m, e = S.sar(kind, phi, 10.0, 2.6e4, 20e3, 7e5, Ms=2.8e5, seeds=16, sig=0.15)
    return dict(kind=kind, phi=phi, SAR=m, se=e)
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(job, [("dilute", None)] + [("loose", x) for x in (0.1, 0.2, 0.3, 0.4, 0.5)])
    json.dump(res, open("../data/dicorato.json", "w"), indent=1)
    for r in res: print(r)
    print("done")
