"""Posterior packing fraction and loss of the EQD gain for low, medium and high perfusion (Table 5); reuses the bank of clin_bayes.py."""
import json, numpy as np, clin_bayes as B
CB = json.load(open("../data/clin_bayes.json")); z = np.load("../data/clin_bayes_bank.npz"); TH, R = z["theta"][:CB["n_bank"]], z["R"][:CB["n_bank"]]
tab = B.sar_table(); g = json.load(open("../data/gain_phi.json"))["clinical"]; meas = np.array(CB["meas"]); wt = B.posterior_S(R, meas).sum(1)
out = {}
for lo, hi in ((0.3e-3, 1e-3), (1e-3, 3e-3), (3e-3, 8.01e-3), (0.3e-3, 8.01e-3)):
    m = (TH[:, 3] >= lo) & (TH[:, 3] < hi); L = B.posterior_S(R[m], meas); rng = np.random.default_rng(3)
    phi, sd = B.phi_posterior(L, 10000, rng, tab); loss = 1 - np.interp(phi, g["phi"], g["gain"]) / g["gain"][0]
    chi = float((((37 + B.S[None, :, None] * R[m][:, None, :3] - meas) / 0.5)**2).sum(-1).min())
    out[f"{lo * 1e3:g}-{hi * 1e3:.0f}"] = dict(n=int(m.sum()), post_mass=float(wt[m].sum() / wt.sum()), phi=np.percentile(phi, [2.5, 50, 97.5]).tolist(),
        SAR=np.percentile(sd, [2.5, 50, 97.5]).tolist(), loss_med=float(np.median(loss)), P_loss50=float((loss > 0.5).mean()), minchi=chi)
    print(lo, hi, out[f"{lo * 1e3:g}-{hi * 1e3:.0f}"])
CB["perfusion_ranges"] = out; json.dump(CB, open("../data/clin_bayes.json", "w"), indent=1, default=float)
