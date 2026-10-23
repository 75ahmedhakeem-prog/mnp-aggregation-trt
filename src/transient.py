"""Session-mean temperature of a 60 min session from the transient Pennes equation (heating from 37 C), solved per Fourier mode:
T_hat(q,t) = Q_hat/L (1 - exp(-L t / rho_c)), L = k q^2 + w rho_b c_b; mean over t in [0, t_s]. Compared with the steady state."""
import json, numpy as np, prostate as P
from uq_therapy import eqd
RHOC = 1045 * 3540.0                     # prostate tissue volumetric heat capacity (J m^-3 K^-1)
def maps(seed, w=3e-3, k=0.5, L=0.12, n=128, ts=3600.0):
    rng = np.random.default_rng(seed); x, X, Y, Z = P.grid(L, n); C = P.depots(rng)
    dep = np.zeros(X.shape, bool)
    for p in C: dep |= (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2 <= 0.0053 ** 2
    Q = np.where(dep, 100.7e3, 0.0); npad = 2 * n; Qp = np.zeros((npad,) * 3); Qp[:n, :n, :n] = Q
    q = 2 * np.pi * np.fft.fftfreq(npad, d=L / n); QX, QY, QZ = np.meshgrid(q, q, q[:npad // 2 + 1], indexing="ij", sparse=True)
    lam = k * (QX ** 2 + QY ** 2 + QZ ** 2) + w * P.RHOC_B; FQ = np.fft.rfftn(Qp)
    a = lam * ts / RHOC; g = 1 - (1 - np.exp(-a)) / a
    Ts = np.fft.irfftn(FQ / lam, s=Qp.shape)[:n, :n, :n]; Tm = np.fft.irfftn(FQ / lam * g, s=Qp.shape)[:n, :n, :n]
    pro = X ** 2 + Y ** 2 + Z ** 2 <= 0.0203 ** 2
    return Ts[pro], Tm[pro], float(1 / (w * P.RHOC_B / RHOC) / 60)
res = {}
for sar in (3.801, 2.015, 0.818):
    rows = []
    for sd in range(8):
        s, m, tau = maps(sd); Ts, Tm = 37 + sar * s, 37 + sar * m
        es, e0 = eqd([Ts] * 6); em, _ = eqd([Tm] * 6)
        rows.append(dict(T90s=np.percentile(Ts, 10), T90m=np.percentile(Tm, 10), g_s=np.percentile(es, 2) - e0, g_m=np.percentile(em, 2) - e0))
    res[str(sar)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}; res["tau_perfusion_min"] = tau
    print(sar, {k: round(v, 2) for k, v in res[str(sar)].items()}, "tau", round(tau, 1), flush=True)
for k in ("3.801", "2.015", "0.818"):
    res[k]["gain_ratio"] = res[k]["g_m"] / res[k]["g_s"]
res["loss01_transient"] = 1 - res["2.015"]["g_m"] / res["3.801"]["g_m"]; res["loss02_transient"] = 1 - res["0.818"]["g_m"] / res["3.801"]["g_m"]
json.dump(res, open("../data/transient.json", "w"), indent=1); print({k: v for k, v in res.items() if k.startswith("loss")})
