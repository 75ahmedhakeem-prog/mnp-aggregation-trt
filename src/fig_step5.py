from plotstyle import *
import json
from scipy.ndimage import gaussian_filter
import clin_bayes as B
CB = json.load(open("../data/clin_bayes.json")); G = json.load(open("../data/gain_phi.json")); CI = json.load(open("../data/clin_inverse.json"))
z = np.load("../data/clin_bayes_bank.npz"); TH, R = z["theta"][:CB["n_bank"]], z["R"][:CB["n_bank"]]
L = B.posterior_S(R, np.array(CB["meas"]))                                   # (draws, S)
fig, axs = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw=dict(wspace=0.30, hspace=0.30))
c, d, a, b = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
# (a) joint posterior of effective SAR and perfusion
se = np.linspace(0.1, 5.0, 50); we = np.linspace(np.log10(0.3), np.log10(8), 31)
Hh, _, _ = np.histogram2d(np.repeat(B.S[None, :], len(TH), 0).ravel(), np.repeat(np.log10(TH[:, 3] * 1e3)[:, None], len(B.S), 1).ravel(), bins=[se, we], weights=L.ravel())
Hh = gaussian_filter(Hh, 1.0); Hh /= Hh.sum(); f = np.sort(Hh.ravel())[::-1]; cf = np.cumsum(f)
lev = [f[np.searchsorted(cf, p)] for p in (0.95, 0.68)]
sc = 0.5 * (se[1:] + se[:-1]); wc = 10 ** (0.5 * (we[1:] + we[:-1]))
a.contour(sc, wc, Hh.T, levels=[lev[0], lev[1]], colors=[BLUE, BLACK], linewidths=[2.0, 2.4], linestyles=["--", "-"])
a.plot([], [], "-", color=BLACK, lw=2.4, label="68% region"); a.plot([], [], "--", color=BLUE, lw=2.0, label="95% region")
fl = CI["sar_table"]["4500"]["0.0"]; a.axvline(fl, color=RED, ls="-.", lw=1.8, label="Fluid SAR (4.5 kA m$^{-1}$)")
a.set_yscale("log"); a.set_yticks([0.3, 1, 3, 8]); a.set_yticklabels(["0.3", "1", "3", "8"]); a.minorticks_off()
a.set_xlim(0, 5); a.set_ylim(0.3, 8); a.set_xlabel("Effective SAR (W g$^{-1}$ Fe)"); a.set_ylabel("Perfusion (10$^{-3}$ s$^{-1}$)")
a.legend(loc="lower right", fontsize=13.5); a.text(0.04, 0.96, "(a)", transform=a.transAxes, fontsize=24, fontweight="bold", va="top")
a.xaxis.set_minor_locator(AutoMinorLocator(2))
# (b) KMC SAR(phi) of the clinical particle with the posterior range of the effective SAR
T = CI["sar_table"]
for H, mk, col in (("4000", "o", GREEN), ("4500", "s", BLUE), ("5000", "^", ORANGE)):
    ph = sorted([p for p in T[H] if float(p) <= 0.3], key=float); b.plot([float(p) for p in ph], [max(T[H][p], 0.0) for p in ph], mk + "-", color=col, ms=9, label=f"{int(H) / 1000:.1f} kA m$^{{-1}}$")
sp = CB["SAR_post"]
for v, ls in ((sp["p2_5"], ":"), (sp["median"], "-"), (sp["p97_5"], ":")): b.axhline(v, color=BLACK, ls=ls, lw=1.6)
b.set_xlabel("Packing fraction, φ"); b.set_ylabel("SAR (W g$^{-1}$ Fe)"); b.set_xlim(-0.01, 0.31); b.set_ylim(0, 5)
b.legend(loc="upper right", fontsize=13.5); b.text(0.04, 0.96, "(b)", transform=b.transAxes, fontsize=24, fontweight="bold", va="top")
b.xaxis.set_minor_locator(AutoMinorLocator(2)); b.yaxis.set_minor_locator(AutoMinorLocator(2))
# (c) EQD gain versus phi
for part, mk, col, lab_ in (("clinical", "s", BLUE, "Clinical, 100 kHz"), ("robust", "o", GREEN, "Robust, 85 kHz")):
    dd = G[part]; c.plot(dd["phi"], dd["gain"], mk + "-", color=col, ms=9, label=lab_)
c.set_xlabel("Packing fraction, φ"); c.set_ylabel("EQD gain at D$_{98}$ (Gy)"); c.set_xlim(-0.01, 0.31); c.set_ylim(0, 9)
c.legend(loc="center right", fontsize=13.5); c.text(0.04, 0.96, "(c)", transform=c.transAxes, fontsize=24, fontweight="bold", va="top")
c.text(0.07, 8.6, "robust: 61 Gy at φ = 0", fontsize=13, color=GREEN, va="top")
c.xaxis.set_minor_locator(AutoMinorLocator(2)); c.yaxis.set_minor_locator(AutoMinorLocator(2))
# (d) posterior of phi
rng = np.random.default_rng(5); tab = B.sar_table(); phi, _ = B.phi_posterior(L, 20000, rng, tab)
d.hist(phi, bins=np.arange(-0.005, 0.31, 0.01), histtype="step", color=BLUE, lw=2.4, weights=np.full(len(phi), 1 / len(phi)))
pp = CB["phi_post"]
for v, ls in ((pp["p2_5"], ":"), (pp["median"], "-"), (pp["p97_5"], ":")): d.axvline(v, color=BLACK, ls=ls, lw=1.6)
d.set_xlabel("Packing fraction, φ"); d.set_ylabel("Posterior probability"); d.set_xlim(-0.01, 0.46); d.set_ylim(0, 0.16)
d.text(0.97, 0.96, f"median {pp['median']:.2f}\n95%: {pp['p2_5']:.2f}–{pp['p97_5']:.2f}\nP(loss > 50%) = {CB['loss_post']['P_gt_0_5']:.2f}",
       transform=d.transAxes, ha="right", va="top", fontsize=14)
d.text(0.04, 0.96, "(d)", transform=d.transAxes, fontsize=24, fontweight="bold", va="top")
d.xaxis.set_minor_locator(AutoMinorLocator(2)); d.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.savefig("../figs/mnp_step5_phi.png"); plt.close(fig); print("ok")
