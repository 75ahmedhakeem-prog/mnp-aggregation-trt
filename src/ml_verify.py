"""KMC verification of the ANN robust design at 50 kHz (16 seeds)."""
import json, numpy as np
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG = 4.77e8, 0.724, 0.15
dsg = json.load(open("../data/ml_design.json"))["robust_50kHz"]
def job(ph):
    f = 5e4; H = LIMIT / f
    m, e = S.sar("dilute" if ph is None else "loose", ph, dsg["d"], dsg["K"], H, f, sig=SIG, seeds=16, cycles=10)
    return dict(phi=ph, SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(job, [None, 0.1, 0.2])
    json.dump(dict(design=dsg, kmc=res), open("../data/ml_verify.json", "w"), indent=1); print(res, "done")
