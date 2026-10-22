from plotstyle import *
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import ml_sar as M
res = json.load(open("../data/ml_sar.json"))
rng = np.random.default_rng(0); idx = rng.permutation(len(M.D)); n = int(0.8 * len(M.D)); tr, te = idx[:n], idx[n:]
m = M.Ens().fit(M.X[tr], M.y[tr]); p = m.predict(M.X[te])
fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw=dict(wspace=0.3))
a = axs[0]; yt = 10 ** M.y[te] - M.SHIFT; yp = 10 ** p - M.SHIFT
a.plot(np.clip(yt, 0.01, None), np.clip(yp, 0.01, None), "o", color=BLUE, ms=7, mfc="white", mew=1.6)
a.plot([0.01, 300], [0.01, 300], "-", color=BLACK, lw=1.6); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.05, 600); a.set_ylim(0.05, 600)
a.set_xlabel("KMC SAR (W g$^{-1}$ Fe)"); a.set_ylabel("ANN SAR (W g$^{-1}$ Fe)")
a.text(0.96, 0.08, f"R$^2$ = {res['random']['ANN']['R2']:.3f}", transform=a.transAxes, ha="right", fontsize=19); lab(a, "(a)")
b = axs[1]; B = res["band"]; x = np.arange(len(B)); w = 0.38
b.bar(x - w / 2, [r["GBT_R2"] for r in B], w, color=GREEN, ec=BLACK, lw=1.1, label="GBT")
b.bar(x + w / 2, [r["ANN_R2"] for r in B], w, color=RED, ec=BLACK, lw=1.1, label="ANN ensemble")
b.axhline(res["noise_ceiling_R2"], color=BLACK, ls="--", lw=1.8, label="Noise limit")
b.set_xticks(x); b.set_xticklabels(["d", "K", "M$_S$", "f", "H", "φ"]); b.set_ylim(0.6, 1.16); b.set_ylabel("Band hold-out R$^2$")
b.legend(loc="upper right", ncol=3, fontsize=14, frameon=False, handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(1.0, 1.0)); b.text(0.03, 0.80, "(b)", transform=b.transAxes, fontsize=26, fontweight="bold")
fig.savefig("../figs/mnp_step3_ml.png"); plt.close(fig); print("ok")
