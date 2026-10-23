"""Infer the effective SAR and perfusion from the clinical prostate temperatures (Johannsen et al. 2007: medians T20 41.1, T50 40.8,
T90 40.1 C) and convert the effective SAR into a packing fraction with the KMC SAR(phi, H) of the clinical particle at 4-5 kA/m."""
import json, numpy as np
from maps_cache import load, key
M = load(); MEAS = np.array([41.1, 40.8, 40.1]); SIGT = 0.5
W = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 8e-3]
Q = {w: np.array([[np.percentile(m, q) for q in (80, 50, 10)] for m in M[key(w, 0.5)]]).mean(0) for w in W}   # rises per W/g Fe
S = np.linspace(0.2, 6.0, 581); Wf = np.linspace(1e-3, 8e-3, 141)
def rise(w): return np.array([np.interp(w, W, [Q[x][j] for x in W]) for j in range(3)])
chi = np.array([[np.sum(((37 + s * rise(w) - MEAS) / SIGT) ** 2) for s in S] for w in Wf])
i, j = np.unravel_index(np.argmin(chi), chi.shape); best = dict(w=float(Wf[i]), SAR=float(S[j]), chi2=float(chi[i, j]))
ok = chi <= chi.min() + 2.30                                  # 68% joint region (2 parameters)
Srange = [float(S[ok.any(0)].min()), float(S[ok.any(0)].max())]; Wrange = [float(Wf[ok.any(1)].min()), float(Wf[ok.any(1)].max())]
print("best", best, "SAR range", Srange, "w range", Wrange)
# SAR(phi, H) of the clinical particle
E = json.load(open("../data/extra_kmc.json")); FS = json.load(open("../data/field_scan.json")); NT = json.load(open("../data/nanotherm_phi.json"))
pts = {}
def add(H, ph, s): pts.setdefault(round(H), {}).setdefault(0.0 if ph is None else round(ph, 2), []).append(s)
for r in E:
    if r["particle"] == "clinical" and r["tag"] in ("clin_H", "phi_curve", "conv_ref") and r["opts"].get("N") is None and "steps" not in r["opts"] and r["opts"].get("cycles") == 10: add(r["H"], r["phi"], r["SAR_Fe"])
for r in FS:
    if r["particle"] == "clinical" and r["f"] == 1e5 and r["x"] == 1.0: add(r["H"], r["phi"], r["SAR_Fe"])
for r in NT:
    if r["H"] in (4000.0, 4500.0, 5000.0): add(r["H"], r["phi"], r["SAR_Fe"])
tab = {H: {ph: float(np.mean(v)) for ph, v in sorted(d.items())} for H, d in sorted(pts.items())}
for H, d in tab.items(): print(H, {p: round(s, 2) for p, s in d.items()})
phi_rng = {}
for H in (4000, 4500, 5000):
    ph = np.array(sorted(tab[H])); s = np.array([tab[H][p] for p in ph]); s = np.minimum.accumulate(s)   # monotone envelope
    fg = np.linspace(ph.min(), ph.max(), 3001); sg = np.interp(fg, ph, s)
    inside = fg[(sg >= Srange[0]) & (sg <= Srange[1])]
    phi_rng[H] = [float(inside.min()), float(inside.max())] if inside.size else None
    phi_rng[f"{H}_best"] = float(fg[np.argmin(np.abs(sg - best["SAR"]))])
print("phi ranges", phi_rng)
json.dump(dict(meas=MEAS.tolist(), sigma_T=SIGT, best=best, SAR_range=Srange, w_range=Wrange, sar_table={str(k): v for k, v in tab.items()}, phi=phi_rng,
               chi=chi.tolist(), S=S.tolist(), W=Wf.tolist()), open("../data/clin_inverse.json", "w"))
