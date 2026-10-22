"""KMC at the stimulation-safe frequency (85 kHz, H*f limit): ANN ridge design, robust 50 kHz design and clinical particle."""
import json, itertools
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG, F = 4.77e8, 0.724, 0.15, 8.5e4
PART = {"ann_safe": (10.7, 3.4e4), "robust50": (14.0, 2.6e4), "clinical": (12.0, 1.4e4)}
def job(a):
    n, ph = a; d, K = PART[n]
    m, e = S.sar("dilute" if ph is None else "loose", ph, d, K, LIMIT / F, F, sig=SIG, seeds=16, cycles=10)
    return dict(particle=n, d=d, K=K, f=F, phi=ph, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(job, list(itertools.product(PART, (None, 0.1, 0.2))), chunksize=1)
    json.dump(res, open("../data/safe_verify.json", "w"), indent=1); print("done")
