import json, numpy as np, time
from multiprocessing import Pool
import sar_cluster as S
FE = 0.724; N = 1200
def sample(i):
    r = np.random.default_rng(1000 + i)
    d = r.uniform(9, 20); K = 10 ** r.uniform(np.log10(6e3), np.log10(4e4)); Ms = r.uniform(3.0e5, 4.8e5)
    f = 10 ** r.uniform(np.log10(2.5e4), np.log10(3e5)); H = 10 ** r.uniform(np.log10(2e3), np.log10(2.5e4))
    phi = 0.0 if r.random() < 0.25 else r.uniform(0.05, 0.30)
    return dict(i=i, d=d, K=K, Ms=Ms, f=f, H=H, phi=phi)
def job(i):
    p = sample(i); kind = "dilute" if p["phi"] == 0 else "loose"
    m, e = S.sar(kind, p["phi"] or None, p["d"], p["K"], p["H"], p["f"], Ms=p["Ms"], sig=0.15, seeds=4, cycles=8)
    return dict(p, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for k, r in enumerate(pool.imap_unordered(job, range(N), chunksize=4)):
            out.append(r)
            if k % 100 == 0: print(k, round(time.time() - t0), flush=True); json.dump(out, open("../data/ml_dataset.json", "w"))
    json.dump(out, open("../data/ml_dataset.json", "w")); print("done", round(time.time() - t0))
