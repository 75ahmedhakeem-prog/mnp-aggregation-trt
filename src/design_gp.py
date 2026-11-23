"""Dense-aggregate optimum (phi = 0.2) chosen by the Gaussian-process baseline (trained on all 1200 KMC samples) on the same
design grids as ml_design.py (50 kHz) and ml_design_safe.py (85 kHz), for comparison with the ANN choice."""
import os, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
import ml_sar as M, ml_baselines as B, design_exact as E
g = B.GP().fit(M.X, M.y); out = {}
for case, c in E.CASES.items():
    f = c["f"]; H = E.LIMIT / f; Dd, KK = np.meshgrid(c["d"], c["K"], indexing="ij")
    X = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(H)), np.full(Dd.size, 0.2)])
    s = 10 ** g.predict(X) - M.SHIFT; i = int(np.argmax(s))
    out[case] = dict(d=float(Dd.ravel()[i]), K=float(KK.ravel()[i]), GP_02=float(s[i]))
json.dump(out, open("../data/design_gp.json", "w"), indent=1); print(out)
# KMC check of the GP choices (16 aggregates, 10 cycles, as design_exact.py)
from multiprocessing import Pool
if __name__ == "__main__":
    with Pool(2) as p: R = p.map(E.kmc, [(c, out[c]["d"], out[c]["K"], 0.2, 16, 10) for c in out])
    for c, r in zip(out, R): out[c].update(KMC_02=r["SAR_Fe"], se_02=r["se_Fe"])
    json.dump(out, open("../data/design_gp.json", "w"), indent=1); print(out)
