import json, itertools, numpy as np, time
from multiprocessing import Pool
import sar_cluster as S
TARGET = {1e5: 58.0, 3e5: 243.0}          # soft aggregates, eta = 83.7 mPa s (Brown suppressed), Fig. 6
def job(a):
    K, d, f = a; m, e = S.sar("loose", 0.10, d, K, 24e3, f, seeds=16)
    return dict(K=K, d=d, f=f, SAR=m, se=e)
if __name__ == "__main__":
    grid = list(itertools.product([1.4e4, 1.8e4, 2.2e4, 2.6e4, 3.0e4], [12.0, 13.0, 14.0], [1e5, 3e5])); t0 = time.time()
    with Pool(2) as p: res = p.map(job, grid, chunksize=1)
    json.dump(res, open("../data/calib_soft.json", "w"), indent=1)
    best = None
    for K, d in itertools.product([1.4e4, 1.8e4, 2.2e4, 2.6e4, 3.0e4], [12.0, 13.0, 14.0]):
        r = {x["f"]: x["SAR"] for x in res if x["K"] == K and x["d"] == d}
        err = sum(np.log(max(r[f], 1e-3) / TARGET[f]) ** 2 for f in TARGET)
        print(K, d, round(r[1e5], 1), round(r[3e5], 1), round(err, 3))
        if best is None or err < best[0]: best = (err, K, d)
    print("best", best, round(time.time() - t0), "s")
