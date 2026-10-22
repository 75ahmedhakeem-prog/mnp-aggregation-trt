import json, numpy as np
from multiprocessing import Pool
import kmc
K, MS, H, F = 1.5e4, 3.75e5, 1500.0, 1e5
def job(d):
    v = [kmc.run(np.zeros((500, 3)), np.full(500, d), K, MS, H, F, dip=False, cycles=16, steps=300, discard=4, seed=s)["SAR"] for s in range(3)]
    return dict(d=d, KMC=float(np.mean(v)), se=float(np.std(v) / np.sqrt(3)), LRT=float(kmc.lrt_sar(d, K, MS, H, F)))
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(job, [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    json.dump(res, open("../data/verify_lrt.json", "w"), indent=1)
    for r in res: print(r)
    print("done")
