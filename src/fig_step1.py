from plotstyle import *
import json
v = json.load(open("../data/verify_lrt.json")); g = json.load(open("../data/validate_ga.json"))
EXP = {1e5: dict(SAR=(29, 6), Hc=(24, 7), Mr=(28, 7)), 3e5: dict(SAR=(19, 8), Hc=(9, 13), Mr=(20, 8))}   # Fig. 8, eta = 83.7 mPa s
PHI_SOFT, PHI_DENSE = 0.10, 0.172
def at(f, key, phi):
    xs = np.array([r["phi"] for r in g if r["f"] == f]); ys = np.array([r[key] for r in g if r["f"] == f]); o = np.argsort(xs)
    return float(np.interp(phi, xs[o], ys[o]))
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.6), gridspec_kw=dict(wspace=0.36, hspace=0.36)); axs = axs.ravel()
a = axs[0]; d = np.array([r["d"] for r in v])
a.plot(d, [r["LRT"] * 1e-3 for r in v], "-", color=BLACK, lw=2.6, label="Linear response")
a.errorbar(d, [r["KMC"] * 1e-3 for r in v], yerr=[2 * r["se"] * 1e-3 for r in v], fmt="o", color=RED, ms=9, mfc="white", mew=2.2, capsize=4, label="KMC (this work)")
a.set_yscale("log"); a.set_xlabel("Core diameter (nm)"); a.set_ylabel("SAR (W g$^{-1}$)"); a.set_xlim(12.5, 20.5); a.set_ylim(5e-3, 6)
a.legend(loc="lower center", fontsize=16); lab(a, "(a)")
b = axs[1]
for f, col, mk in ((1e5, BLUE, "o"), (3e5, RED, "s")):
    xs = np.array(sorted(r["phi"] for r in g if r["f"] == f)); s0 = at(f, "SAR", PHI_SOFT)
    b.plot(xs, [at(f, "SAR", x) / s0 for x in xs], "-" + mk, color=col, lw=2.4, ms=8, label=f"Model, {int(f/1e3)} kHz")
    m, e = EXP[f]["SAR"]; b.errorbar([PHI_DENSE], [1 - m / 100], yerr=[e / 100], fmt="*", color=col, ms=19, mec=BLACK, capsize=5, zorder=5)
b.plot([], [], "*", color=GREY, ms=17, mec=BLACK, label="Experiment (dense)")
b.set_xlabel("Packing fraction, φ"); b.set_ylabel("SAR / SAR$_{soft}$"); b.set_ylim(0.2, 1.25); b.set_xlim(0.09, 0.26)
b.legend(loc="upper right", fontsize=15); lab(b, "(b)", y=0.08, va="bottom") if False else b.text(0.04, 0.05, "(b)", transform=b.transAxes, fontsize=19*1.35, fontweight="bold")
c = axs[2]; keys = ["SAR", "Mr", "Hc"]; x = np.arange(6); w = 0.38
mod = [100 * (1 - at(f, k, PHI_DENSE) / at(f, k, PHI_SOFT)) for f in (1e5, 3e5) for k in keys]
ex = [EXP[f][k][0] for f in (1e5, 3e5) for k in keys]; ee = [EXP[f][k][1] for f in (1e5, 3e5) for k in keys]
c.bar(x - w / 2, ex, w, yerr=ee, color=GREY, ec=BLACK, lw=1.2, capsize=4, label="Experiment")
c.bar(x + w / 2, mod, w, color=RED, ec=BLACK, lw=1.2, label="Model")
c.axhline(0, color=BLACK, lw=1.2); c.axvline(2.5, color=BLACK, ls="--", lw=1.2)
c.set_xticks(x); c.set_xticklabels(["SAR", "M$_R$", "H$_C$"] * 2, fontsize=17); c.set_ylabel("Decrease, dense vs soft (%)"); c.set_ylim(-15, 56)
c.text(1.0, 46, "100 kHz", ha="center", fontsize=19); c.text(4.0, 46, "300 kHz", ha="center", fontsize=19)
c.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=15); c.text(0.04, 0.05, "(c)", transform=c.transAxes, fontsize=19*1.35, fontweight="bold")
dd = json.load(open("../data/dicorato.json")); e4 = axs[3]
dil = [r for r in dd if r["kind"] == "dilute"][0]; cl = [r for r in dd if r["kind"] == "loose"]
e4.errorbar([0.0], [dil["SAR"]], yerr=[2 * dil["se"]], fmt="o", color=BLUE, ms=10, capsize=4, label="Model, dispersed")
e4.errorbar([r["phi"] for r in cl], [r["SAR"] for r in cl], yerr=[2 * r["se"] for r in cl], fmt="-o", color=RED, ms=9, lw=2.4, capsize=4, label="Model, clustered")
e4.plot([0.0], [230], "*", color=GREY, ms=20, mec=BLACK, label="Exp., solution")
e4.plot([0.45], [110], "*", color=WHITE if False else "#FFFFFF", ms=20, mec=BLACK, mew=2, label="Exp., in cells")
e4.set_xlabel("Packing fraction, φ"); e4.set_ylabel("SAR (W g$^{-1}$)"); e4.set_xlim(-0.04, 0.55); e4.set_ylim(0, 320)
e4.legend(loc="upper right", fontsize=15, ncol=1); e4.text(0.04, 0.05, "(d)", transform=e4.transAxes, fontsize=19*1.35, fontweight="bold")
fig.savefig("../figs/mnp_step1_sar.png"); plt.close(fig)
print({f"{int(f/1e3)}kHz": [round(100 * (1 - at(f, k, PHI_DENSE) / at(f, k, PHI_SOFT)), 1) for k in keys] for f in (1e5, 3e5)})
