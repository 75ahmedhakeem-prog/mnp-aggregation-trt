"""(a) progressive aggregation across six sessions matched to the clinical session drop; (b) field needed for T90 = 41 C;
(c) thermoradiotherapy at the stimulation-safe frequency (85 kHz)."""
import json, numpy as np
from maps_cache import load, key
from uq_therapy import eqd
import stim
M = load(); mp = M[key(3e-3, 0.5)]
m90 = float(np.mean([np.percentile(m, 10) for m in mp]))           # T90 rise per W/g Fe
FS = json.load(open("../data/field_scan.json")); SV = json.load(open("../data/safe_verify.json"))
res = dict(T90_rise_per_SAR=m90, SAR_for_T90_41=4.0 / m90)
def gain(sars):     # list of six session SAR values -> mean D98 gain over layouts
    out = []
    for m in mp:
        e, e0 = eqd([37.0 + s * m for s in sars]); out.append(np.percentile(e, 2) - e0)
    return float(np.mean(out))
# ---------------- (a) progressive aggregation
pts = [(0.0 if r["phi"] is None else r["phi"], r["SAR_Fe"], r["se_Fe"]) for r in FS if r["particle"] == "clinical" and r["f"] == 1e5 and r["x"] == 1.0]
# the 16-seed set and the 12-seed set both exist at x = 1; average duplicates
from collections import defaultdict
acc = defaultdict(list)
for ph, s, e in pts: acc[ph].append(s)
PH = np.array(sorted(acc)); SAR = np.array([np.mean(acc[p]) for p in PH])
sar_phi = lambda ph: float(np.interp(ph, PH, SAR))
target = 0.62 * SAR[0]
phis = np.linspace(0, 0.2, 2001); phi6 = float(phis[np.argmin(np.abs([sar_phi(p) - target for p in phis]))])
sess = lambda p6: [sar_phi(p6 * k / 5) for k in range(6)]
res["progressive"] = dict(phi_points=PH.tolist(), SAR_points=SAR.tolist(), phi6_matched=phi6,
    sessions_matched=sess(phi6), gain_matched=gain(sess(phi6)), gain_fluid_all=gain([SAR[0]] * 6),
    sessions_to02=sess(0.2), gain_to02=gain(sess(0.2)), gain_const01=gain([sar_phi(0.1)] * 6), gain_const02=gain([sar_phi(0.2)] * 6),
    T90_sessions_matched=[37 + s * m90 for s in sess(phi6)], T90_sessions_to02=[37 + s * m90 for s in sess(0.2)])
print("progressive", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["progressive"].items() if not isinstance(v, list)})
# ---------------- (b) required field ratio x = (H f)/limit for T90 = 41 C
req = []
for part in ("clinical", "robust"):
    for f in (5e4, 7e4, 1e5, 1.4e5, 2e5):
        for ph in (None, 0.1, 0.2):
            rows = sorted([r for r in FS if r["particle"] == part and r["f"] == f and r["phi"] == ph], key=lambda r: r["x"])
            xs = {}; [xs.setdefault(r["x"], []).append(r["SAR_Fe"]) for r in rows]
            X = np.array(sorted(xs)); S_ = np.array([np.mean(xs[x]) for x in X]); sreq = 4.0 / m90
            if S_.max() < sreq: xr = float("inf")
            else:
                j = int(np.argmax(S_ >= sreq))
                if j == 0: xr = float(X[0])
                else:
                    a, b = max(S_[j - 1], 1e-3), S_[j]; xr = float(np.exp(np.interp(np.log(sreq), [np.log(a), np.log(b)], [np.log(X[j - 1]), np.log(X[j])])))
            sl = float(np.mean(xs.get(1.0, [np.nan]))); req.append(dict(particle=part, f=f, phi=ph, x_req=xr, SAR_at_limit=sl, gain_at_limit=gain([max(sl, 0.0)] * 6), T90_at_limit=37 + max(sl, 0.0) * m90, stim_at_limit=float(stim.index(f)), stim_at_req=float(stim.index(f, xr * stim.LIMIT / f)) if np.isfinite(xr) else None))
res["required"] = req
for r in req: print("req", r["particle"], int(r["f"] / 1e3), r["phi"], round(r["x_req"], 2), round(r["SAR_at_limit"], 2), r["stim_at_req"] and round(r["stim_at_req"], 2))
# ---------------- (c) therapy at 85 kHz
saf = {}
for r in SV:
    nm = f'{r["particle"]}|{"fluid" if r["phi"] is None else r["phi"]}'
    saf[nm] = dict(SAR=r["SAR_Fe"], se=r["se_Fe"], gain=gain([max(r["SAR_Fe"], 0)] * 6), T90=37 + max(r["SAR_Fe"], 0) * m90)
res["safe85"] = dict(f=8.5e4, stim_index=float(stim.index(8.5e4)), f_min=float(stim.fmin()), scen=saf)
for k, v in saf.items(): print("85k", k, {q: round(x, 2) for q, x in v.items()})
json.dump(res, open("../data/step4_extra.json", "w"), indent=1); print("done")
