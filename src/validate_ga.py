import json, numpy as np, time
from multiprocessing import Pool
import sar_cluster as S
import re
best = [l for l in open('../data/calib_soft.log') if l.startswith('best')][0]
K, D = [float(x) for x in re.findall(r'\), ([0-9.e+]+), ([0-9.e+]+)\)', best)[0]]
SEEDS = 40
def job(a):
    phi, f = a; sars, loops = S.sar("loose", phi, D, K, 24e3, f, seeds=SEEDS, full=True)
    hm = np.array([S.hc_mr(l) for l in loops])
    return dict(phi=phi, f=f, SAR=float(np.mean(sars)), se=float(np.std(sars) / np.sqrt(SEEDS)),
                Hc=float(np.mean(hm[:, 0])), Mr=float(np.mean(hm[:, 1])), sars=sars)
if __name__ == "__main__":
    PHI = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]; t0 = time.time()
    with Pool(2) as p: res = p.map(job, [(ph, f) for ph in PHI for f in (1e5, 3e5)], chunksize=1)
    json.dump(res, open("../data/validate_ga.json", "w"), indent=1)
    for r in res: print(r["phi"], int(r["f"] / 1e3), round(r["SAR"], 1), round(r["se"], 1), round(r["Hc"]), round(r["Mr"]))
    print("done", round(time.time() - t0))
