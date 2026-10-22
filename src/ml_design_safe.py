"""ANN design search at the lowest stimulation-safe frequency (85 kHz, index < 1) and at the H*f limit."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import ml_sar as M, stim
m = M.Ens().fit(M.X, M.y)
f = 8.5e4; H = stim.LIMIT / f
d = np.linspace(9.5, 18, 86); K = np.logspace(np.log10(8e3), np.log10(4e4), 81); Dd, KK = np.meshgrid(d, K, indexing="ij")
def pred(phi):
    X = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(H)), np.full(Dd.size, phi)])
    return 10 ** m.predict(X) - M.SHIFT
s0, s1, s2 = pred(0.0), pred(0.1), pred(0.2); i = int(np.argmax(s2))
out = dict(f=f, H=H, stim_index=float(stim.index(f)), d=float(Dd.ravel()[i]), K=float(KK.ravel()[i]), ann=dict(disp=float(s0[i]), phi01=float(s1[i]), phi02=float(s2[i])))
json.dump(out, open("../data/ml_design_safe.json", "w"), indent=1); print(out)
