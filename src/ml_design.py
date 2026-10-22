"""Use the ANN ensemble (trained on all 1200 KMC samples) to search the design space at the safety limit and compare with the KMC design map."""
import json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import ml_sar as M
m = M.Ens().fit(M.X, M.y)
LIMIT = 4.77e8
d = np.linspace(10, 18, 81); K = np.logspace(np.log10(8e3), np.log10(3.4e4), 81)
out = {}
for f in (5e4, 1e5):
    H = LIMIT / f
    for phi in (0.0, 0.1, 0.2):
        Dd, KK = np.meshgrid(d, K, indexing="ij")
        X = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(H)), np.full(Dd.size, phi)])
        t0 = time.time(); s = 10 ** m.predict(X) - M.SHIFT; dt = time.time() - t0
        # robust objective: mean SAR over phi = 0.1 and 0.2 is evaluated separately below
        i = int(np.argmax(s)); out[f"{int(f/1e3)}kHz_phi{phi}"] = dict(d=float(X[i, 0]), K=float(10 ** X[i, 1]), SAR=float(s[i]), n=int(Dd.size), sec=dt)
# robust design: maximise the SAR of the phi = 0.2 state (worst case) at 50 kHz, report all states
f = 5e4; H = LIMIT / f; Dd, KK = np.meshgrid(d, K, indexing="ij")
def pred(phi):
    X = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(H)), np.full(Dd.size, phi)])
    return 10 ** m.predict(X) - M.SHIFT
s0, s1, s2 = pred(0.0), pred(0.1), pred(0.2); i = int(np.argmax(s1 + s2))
out["robust_50kHz"] = dict(d=float(Dd.ravel()[i]), K=float(KK.ravel()[i]), SAR_disp=float(s0[i]), SAR_01=float(s1[i]), SAR_02=float(s2[i]))
json.dump(out, open("../data/ml_design.json", "w"), indent=1)
for k, v in out.items(): print(k, {q: round(x, 3) if isinstance(x, float) else x for q, x in v.items()})
