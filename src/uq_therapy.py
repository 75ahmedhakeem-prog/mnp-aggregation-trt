"""Uncertainty, sensitivity and tumour control analysis of the prostate thermoradiotherapy scenarios.
(1) 95% intervals of the D98 EQD gain from the KMC standard errors of SAR and the eight depot layouts;
(2) one-at-a-time sensitivity (tornado) of the gain and of the fraction of the planned gain lost by aggregation;
(3) Poisson TCP with inter-patient spread of alpha, calibrated to 86% control after 70 Gy in 28 fractions (RTOG 0415)."""
import json, numpy as np
from trbio import PAR, lq, k_heat
from maps_cache import load, key
M = load(); VOX = (0.12 / 128) ** 3 * 1e6          # cm^3
BASE = dict(PAR["SiHa"], a37=0.15, ab37=1.5, mu=0.5, rb=1.0)
SC = {"fluid": (3.801, 0.585), "c01": (2.015, 0.410), "c02": (0.818, 0.387),
      "c50_01": (3.211, 0.753), "c50_02": (1.164, 0.453), "r01": (5.768, 0.799), "r02": (4.377, 0.511)}
def eqd(Tlist, p=BASE, d=2.5, nfx=28, tint=0.0, t_ht=3600.0, direct=True):
    a37 = p["a37"]; b37 = a37 / p["ab37"]; den = a37 + 2 * b37
    e_rt = (a37 * d + b37 * d * d) / den; tot = (nfx - len(Tlist)) * e_rt
    for T in Tlist:
        a, b = lq(T, tint, p); e = (a * d + b * d * d) / den
        if direct: e = e + k_heat(T, p) * t_ht / den
        tot = tot + e
    return tot, nfx * e_rt
def gain98(sar, mp, **kw):
    out = []
    for m in mp:
        T = 37.0 + sar * m; e, e0 = eqd([T] * 6, **kw); out.append(np.percentile(e, 2) - e0)
    return float(np.mean(out))
res = {}
# ---------------- (1) intervals
rng = np.random.default_rng(1); NS = 300; mp = M[key(3e-3, 0.5)]
samp = {}
for s, (mu, se) in SC.items():
    v = []
    for i in range(NS):
        sar = max(rng.normal(mu, se), 0.0); m = mp[rng.integers(8)]
        e, e0 = eqd([37.0 + sar * m] * 6); v.append(np.percentile(e, 2) - e0)
    samp[s] = np.array(v)
ci = {s: dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))) for s, v in samp.items()}
for s in ("c01", "c02"):
    loss = 1 - samp[s] / samp["fluid"]; ci["loss_" + s] = dict(mean=float(np.mean(loss)), lo=float(np.percentile(loss, 2.5)), hi=float(np.percentile(loss, 97.5)))
res["ci"] = ci; print("CI", {k: {q: round(x, 2) for q, x in v.items()} for k, v in ci.items()}, flush=True)
# ---------------- (2) sensitivity
VAR = {"base": dict(),
       "perfusion 2e-3": dict(maps=key(2e-3, 0.5)), "perfusion 5e-3": dict(maps=key(5e-3, 0.5)),
       "k 0.4": dict(maps=key(3e-3, 0.4)), "k 0.6": dict(maps=key(3e-3, 0.6)),
       "alpha/beta 3 Gy": dict(p=dict(BASE, ab37=3.0)), "alpha37 0.10": dict(p=dict(BASE, a37=0.10)), "alpha37 0.30": dict(p=dict(BASE, a37=0.30)),
       "HeLa thermal ratio": dict(p=dict(BASE, ra=PAR["HeLa"]["ra"])), "beta(T) as SiHa": dict(p=dict(BASE, rb=PAR["SiHa"]["rb"])),
       "interval 1 h": dict(tint=1.0), "no direct kill": dict(direct=False), "45 min sessions": dict(t_ht=2700.0)}
sens = {}
for nm, v in VAR.items():
    mp_ = M[v.get("maps", key(3e-3, 0.5))]; kw = {k: v[k] for k in ("p", "tint", "direct", "t_ht") if k in v}
    g = {s: gain98(SC[s][0], mp_, **kw) for s in ("fluid", "c01", "c02", "r01", "r02")}
    g["loss01"] = 1 - g["c01"] / g["fluid"]; g["loss02"] = 1 - g["c02"] / g["fluid"]; sens[nm] = g
    print(nm, {k: round(x, 2) for k, x in g.items()}, flush=True)
res["sens"] = sens
# ---------------- (3) TCP
def run_tcp(sig):
    global F, LR
    F = np.random.default_rng(7).normal(1.0, sig, 400).clip(0.05)
    def tcp(e, logrho, p=BASE):
        a37 = p["a37"]; b37 = a37 / p["ab37"]
        lnN = logrho * np.log(10) + np.log(VOX)
        s = np.array([np.exp(lnN + np.log(np.sum(np.exp(-(f * a37 + 2 * b37) * e - 0.0)))) for f in F])
        return float(np.mean(np.exp(-s)))
    m0 = mp[0]; e_rt, _ = eqd([np.full(m0.shape, 37.0)] * 0)
    e_rt = np.full(m0.shape, 28 * (0.15 * 2.5 + 0.1 * 6.25) / 0.35)
    lo, hi = 3.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tcp(e_rt, mid) > 0.86: lo = mid
        else: hi = mid
    LR = 0.5 * (lo + hi); print("log10 clonogen density (cm^-3)", round(LR, 2), flush=True)
    def iso_dose(t):   # RT-alone dose in 2.5 Gy fractions giving the same TCP
        a, b = 40.0, 110.0
        for _ in range(50):
            mid = 0.5 * (a + b); e = np.full(m0.shape, mid / 2.5 * (0.15 * 2.5 + 0.1 * 6.25) / 0.35)
            if tcp(e, LR) < t: a = mid
            else: b = mid
        return 0.5 * (a + b)
    T = {}
    for s, (mu, se) in SC.items():
        v = []
        for m in mp:
            e, _ = eqd([37.0 + mu * m] * 6); v.append(tcp(e, LR))
        t = float(np.mean(v)); T[s] = dict(TCP=t, iso_dose=iso_dose(t))
        print("TCP", s, round(t, 3), "iso RT dose", round(T[s]["iso_dose"], 1), flush=True)
    return dict(log10_rho=LR, sigma=sig, rt_alone=0.86, scen=T)
res["tcp"] = {str(sg): run_tcp(sg) for sg in (0.10, 0.15, 0.25)}
json.dump(res, open("../data/uq_therapy.json", "w"), indent=1); print("done")
