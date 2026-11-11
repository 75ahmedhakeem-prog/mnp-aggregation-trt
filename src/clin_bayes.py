"""Bayesian inference of the packing fraction phi in patients from the clinical prostate temperatures
(Johannsen et al. 2007: median T20 41.1, T50 40.8, T90 40.1 C).
Nuisance parameters are sampled from their priors and marginalised: perfusion w (0.3-8 x 1e-3 1/s, log-uniform),
tissue conductivity k (0.4-0.6 W/m/K), number (12-36) and radius (4-7 mm) of the depots at a fixed injected iron mass,
the iron mass itself (x0.8-1.2), the field (4-5 kA/m) and the KMC standard errors of SAR(phi).
Also: identifiability test with synthetic patients of known phi, and the posterior loss of the EQD gain."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import prostate as P
MEAS = np.array([41.1, 40.8, 40.1]); SIGT = 0.5
C0 = 100.7e3; R0 = 0.0053; N0 = 24; V0 = N0 * R0**3                      # nominal depot set (prostate.py)
S = np.linspace(0.02, 6.0, 1200)

def forward(th):
    """temperature-rise percentiles (T20, T50, T90) per W/g Fe for one nuisance draw"""
    n, r, k, w, fm, seed = th
    rng = np.random.default_rng(seed); C = P.depots(rng, n_dep=n, r_dep=r, dmin=1.6 * r)
    c_fe = C0 * fm * V0 / (len(C) * r**3)                                  # same injected iron mass in all layouts
    x, X, Y, Z = P.grid(); dep = np.zeros(X.shape, bool)
    for p in C: dep |= (X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2 <= r**2
    T = P.solve(np.where(dep, c_fe, 0.0), 0.12, k=k, w=w); pro = X**2 + Y**2 + Z**2 <= 0.0203**2
    t = T[pro]; return [float(np.percentile(t, q)) for q in (80, 50, 10)] + [len(C)]

def prior(rng, m):
    return [(int(rng.integers(12, 37)), float(rng.uniform(0.004, 0.007)), float(rng.uniform(0.4, 0.6)),
             float(np.exp(rng.uniform(np.log(0.3e-3), np.log(8e-3)))), float(rng.uniform(0.8, 1.2)), int(rng.integers(1e9))) for _ in range(m)]

def posterior_S(R, meas, sig=SIGT):
    chi = (((37 + S[None, :, None] * R[:, None, :3] - meas) / sig)**2).sum(-1)    # (draws, S)
    L = np.exp(-0.5 * (chi - chi.min())); return L                                # unnormalised, per nuisance draw

def sar_table():
    E = json.load(open("../data/extra_kmc.json")); FS = json.load(open("../data/field_scan.json")); NT = json.load(open("../data/nanotherm_phi.json"))
    pts = {}
    def add(H, ph, s, se): pts.setdefault(round(H), {}).setdefault(0.0 if ph is None else round(ph, 2), []).append((s, se if se else 0.3))
    for r in E:
        if r["particle"] == "clinical" and r["tag"] in ("clin_H", "phi_curve", "conv_ref") and r["opts"].get("N") is None and "steps" not in r["opts"] and r["opts"].get("cycles") == 10: add(r["H"], r["phi"], r["SAR_Fe"], r.get("se_Fe"))
    for r in FS:
        if r["particle"] == "clinical" and r["f"] == 1e5 and r["x"] == 1.0: add(r["H"], r["phi"], r["SAR_Fe"], r.get("se_Fe"))
    for r in NT:
        if r["H"] in (4000.0, 4500.0, 5000.0): add(r["H"], r["phi"], r["SAR_Fe"], r.get("se_Fe"))
    tab = {}
    for H in (4000, 4500, 5000):
        ph = sorted(p for p in pts[H] if p <= 0.3)
        m = np.array([np.mean([a for a, _ in pts[H][p]]) for p in ph]); se = np.array([np.sqrt(np.sum([b**2 for _, b in pts[H][p]])) / len(pts[H][p]) for p in ph])
        tab[H] = (np.array(ph), m, se)
    return tab

def sar_to_phi(s, H, tab, rng):
    """invert a noisy, monotone KMC curve SAR(phi) at field H (linear in H between 4.0, 4.5 and 5.0 kA/m)"""
    Hs = sorted(tab); i = min(int(np.searchsorted(Hs, H)), len(Hs) - 1); i = max(i, 1); a, b = Hs[i - 1], Hs[i]; t = (H - a) / (b - a)
    ph = tab[a][0]; curve = (1 - t) * (tab[a][1] + rng.normal(0, 1, len(ph)) * tab[a][2]) + t * (tab[b][1] + rng.normal(0, 1, len(ph)) * tab[b][2])
    curve = np.minimum.accumulate(curve); fg = np.linspace(0, ph.max(), 601); sg = np.interp(fg, ph, curve)
    if s >= sg[0]: return 0.0
    if s <= sg[-1]: return float(ph.max())
    return float(fg[np.argmax(sg <= s)])

def phi_posterior(L, ndraw, rng, tab):
    ps = L.sum(0); ps = ps / ps.sum(); sd = rng.choice(S, size=ndraw, p=ps)
    return np.array([sar_to_phi(s, rng.uniform(4000, 5000), tab, rng) for s in sd]), sd

if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 480
    rng = np.random.default_rng(2026); TH = prior(rng, nb + 15); t0 = time.time()
    with Pool(2) as pool: R = np.array(pool.map(forward, TH, chunksize=4))
    print("forward runs", len(R), round(time.time() - t0), "s", flush=True)
    TH = np.array(TH, dtype=float); bank, Rb, synth, Rs = TH[:nb], R[:nb], TH[nb:], R[nb:]
    tab = sar_table(); out = {"meas": MEAS.tolist(), "sigma_T": SIGT, "n_bank": nb}
    # ---- clinical posterior
    L = posterior_S(Rb, MEAS); wpost = L.sum(1); wpost = wpost / wpost.sum()
    phi, sd = phi_posterior(L, 20000, rng, tab)
    q = lambda v, p: float(np.percentile(v, p))
    out["SAR_post"] = dict(median=q(sd, 50), p2_5=q(sd, 2.5), p16=q(sd, 16), p84=q(sd, 84), p97_5=q(sd, 97.5))
    out["phi_post"] = dict(median=q(phi, 50), p2_5=q(phi, 2.5), p16=q(phi, 16), p84=q(phi, 84), p97_5=q(phi, 97.5),
                           P_ge_0_05=float((phi >= 0.05).mean()), P_ge_0_10=float((phi >= 0.10).mean()), P_fluid=float((phi == 0).mean()))
    ww = bank[:, 3]; o = np.argsort(ww); cw = np.cumsum(wpost[o])
    out["perf_post"] = dict(p2_5=float(ww[o][np.searchsorted(cw, 0.025)]), median=float(ww[o][np.searchsorted(cw, 0.5)]), p97_5=float(ww[o][np.searchsorted(cw, 0.975)]))
    out["perf_post_mass_below_1e-3"] = float(wpost[ww < 1e-3].sum())
    # fluid SAR at the tolerated field: posterior predictive check of the fluid hypothesis
    s_fluid = np.array([tab[H][1][0] for H in (4000, 4500, 5000)])
    chi_f = [float((((37 + s * Rb[:, :3] - MEAS) / SIGT)**2).sum(1).min()) for s in s_fluid]
    out["fluid_best_chi2"] = dict(zip(["4.0", "4.5", "5.0"], chi_f)); out["post_best_chi2"] = float((((37 + S[None, :, None] * Rb[:, None, :3] - MEAS) / SIGT)**2).sum(-1).min())
    # loss of the EQD gain (clinical particle, 100 kHz, 4.77 kA/m)
    g = json.load(open("../data/gain_phi.json"))["clinical"]; loss = 1 - np.interp(phi, g["phi"], g["gain"]) / g["gain"][0]
    out["loss_post"] = dict(median=q(loss, 50), p2_5=q(loss, 2.5), p97_5=q(loss, 97.5), P_gt_0_5=float((loss > 0.5).mean()), P_gt_0_3=float((loss > 0.3).mean()))
    print(json.dumps({k: out[k] for k in ("SAR_post", "phi_post", "perf_post", "perf_post_mass_below_1e-3", "fluid_best_chi2", "post_best_chi2", "loss_post")}, default=lambda v: round(v, 3)), flush=True)
    # ---- identifiability: synthetic patients with known phi (H = 4.5 kA/m), noise 0.3 C on each percentile
    out["synthetic"] = []
    for j, (th, r) in enumerate(zip(synth, Rs)):
        ph_true = [0.0, 0.05, 0.10, 0.15, 0.20][j % 5]; s_true = float(np.interp(ph_true, tab[4500][0], tab[4500][1]))
        meas = 37 + s_true * r[:3] + rng.normal(0, 0.3, 3)
        Ls = posterior_S(Rb, meas); ph, _ = phi_posterior(Ls, 4000, rng, tab)
        rec = dict(phi_true=ph_true, w_true=float(th[3]), median=q(ph, 50), p2_5=q(ph, 2.5), p97_5=q(ph, 97.5), T=meas.tolist())
        rec["covered"] = bool(rec["p2_5"] <= ph_true <= rec["p97_5"]); out["synthetic"].append(rec)
        print("synthetic", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "T"}, flush=True)
    out["coverage_95"] = float(np.mean([r["covered"] for r in out["synthetic"]]))
    np.savez_compressed("../data/clin_bayes_bank.npz", theta=TH, R=R)
    json.dump(out, open("../data/clin_bayes.json", "w"), indent=1, default=float); print("coverage", out["coverage_95"])
