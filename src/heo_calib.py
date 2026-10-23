"""Prostate-specific thermal radiosensitisation calibrated on Heo et al. (2024): 2 Gy + 30 min heating, EQD (iso-effective single dose)
HTRT = 9.47/1.26 = 7.52 Gy (PC3), 4.71/1.20 = 3.93 Gy (LNCaP); MHRT = 9.47 / 4.71 Gy. Heating 30 min at 42-44 C (>42 C, <45 C; Heo et al. 2022 abstract),
interval 0.5 h. Cell LQ parameters from Carlson et al. (2004): PC3 alpha 0.145, a/b 4.1; LNCaP alpha 0.351, a/b 1.09. Direct kill with SiHa Arrhenius.
The fitted alpha ratio (41 vs 37 C) is then used in the prostate planning model."""
import json, numpy as np
from scipy.optimize import brentq
from trbio import PAR, lq, k_heat
from maps_cache import load, key
from uq_therapy import eqd, BASE
CELL = {"PC3": dict(a37=0.145, ab37=4.105, htrt=9.47 / 1.26, mhrt=9.47), "LNCaP": dict(a37=0.351, ab37=1.086, htrt=4.71 / 1.20, mhrt=4.71)}
def eqd_single(ra, c, T, tint=0.5, d=2.0, t_ht=1800.0):
    p = dict(PAR["SiHa"], a37=c["a37"], ab37=c["ab37"], ra=ra, rb=1.0, mu=0.5); b37 = c["a37"] / c["ab37"]
    a, b = lq(T, tint, p); E = a * d + b * d * d + k_heat(T, p) * t_ht
    return (-c["a37"] + np.sqrt(c["a37"] ** 2 + 4 * b37 * E)) / (2 * b37)        # iso-effective single dose
M = load(); mp = M[key(3e-3, 0.5)]
def gains(ra):
    p = dict(BASE, ra=ra); out = {}
    for nm, s in (("fluid", 3.801), ("c01", 2.015), ("c02", 0.818)):
        out[nm] = float(np.mean([(lambda e: np.percentile(e[0], 2) - e[1])(eqd([37 + s * m] * 6, p=p)) for m in mp]))
    out["loss01"] = 1 - out["c01"] / out["fluid"]; out["loss02"] = 1 - out["c02"] / out["fluid"]; return out
res = {}
for cl, c in CELL.items():
    for T in (42.0, 43.0, 44.0):
        for lab, tgt in (("HTRT", c["htrt"]), ("MHRT", c["mhrt"])):
            f = lambda ra: eqd_single(ra, c, T) - tgt
            try: ra = brentq(f, 1.0, 200.0)
            except ValueError: ra = float("nan")
            g = gains(ra) if np.isfinite(ra) else {}
            res[f"{cl}|{T}|{lab}"] = dict(ra=ra, target=tgt, direct_only=float(eqd_single(1.0, c, T)), **g)
            print(cl, T, lab, "ra", round(ra, 3), "EQD at ra=1", round(eqd_single(1.0, c, T), 2), {k: round(v, 2) for k, v in g.items()}, flush=True)
json.dump(res, open("../data/heo_calib.json", "w"), indent=1)
