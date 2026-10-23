"""EQD gain versus packing fraction (clinical particle at 100 kHz / 4.77 kA/m; robust particle at 85 kHz on the limit),
half-loss packing fraction, mixed dispersed/aggregated states, and KMC convergence."""
import json, numpy as np
from maps_cache import load, key
from uq_therapy import eqd
M = load(); mp = M[key(3e-3, 0.5)]
def gain(s): return float(np.mean([np.percentile(eqd([37 + max(s, 0) * m] * 6)[0], 2) - 80.0 for m in mp]))
E = json.load(open("../data/extra_kmc.json")); FS = json.load(open("../data/field_scan.json")); SV = json.load(open("../data/safe_verify.json"))
cur = {"clinical": {}, "robust": {}}
for r in FS:
    if r["particle"] == "clinical" and r["f"] == 1e5 and r["x"] == 1.0: cur["clinical"].setdefault(0.0 if r["phi"] is None else r["phi"], []).append(r["SAR_Fe"])
for r in SV:
    if r["particle"] == "robust50": cur["robust"].setdefault(0.0 if r["phi"] is None else r["phi"], []).append(r["SAR_Fe"])
for r in E:
    if r["tag"] == "phi_curve": cur[r["particle"]].setdefault(r["phi"], []).append(r["SAR_Fe"])
    if r["tag"] == "conv_ref": cur["clinical"].setdefault(r["phi"], []).append(r["SAR_Fe"])
out = {}
for part, d in cur.items():
    ph = sorted(d); s = [float(np.mean(d[p])) for p in ph]; g = [gain(x) for x in s]
    fg = np.linspace(0, max(ph), 301); gg = np.interp(fg, ph, g); half = float(fg[np.argmax(gg <= 0.5 * g[0])]) if (gg <= 0.5 * g[0]).any() else None
    out[part] = dict(phi=ph, SAR=s, gain=g, half_loss_phi=half)
    print(part, [(p, round(a, 2), round(b, 2)) for p, a, b in zip(ph, s, g)], "half-loss phi", half)
# mixed states: fraction x of particles still dispersed, rest at phi = 0.2 (clinical, 100 kHz)
c = out["clinical"]; s0 = c["SAR"][0]; s2 = float(np.interp(0.2, c["phi"], c["SAR"]))
out["mixed"] = [dict(x_dispersed=x, SAR=x * s0 + (1 - x) * s2, gain=gain(x * s0 + (1 - x) * s2)) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
print("mixed", [(m["x_dispersed"], round(m["gain"], 2)) for m in out["mixed"]])
# convergence
conv = {}
for r in E:
    if r["tag"].startswith("conv"): conv.setdefault(r["tag"], {})[r["phi"]] = (r["SAR_Fe"], r["se_Fe"])
out["convergence"] = {t: {str(p): v for p, v in d.items()} for t, d in conv.items()}
for t, d in conv.items(): print(t, {p: (round(a, 2), round(b, 2)) for p, (a, b) in d.items()})
json.dump(out, open("../data/gain_phi.json", "w"), indent=1)
