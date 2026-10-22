from plotstyle import *
import json
nf = json.load(open("../data/nt_field.json")); ps = json.load(open("../data/prostate_sweep.json"))
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.6), gridspec_kw=dict(wspace=0.36, hspace=0.36)); axs = axs.ravel()
cols = {None: BLACK, 0.1: BLUE, 0.2: GREEN, 0.3: RED}; mks = {None: "o", 0.1: "s", 0.2: "^", 0.3: "D"}
a = axs[0]
for ph in (None, 0.1, 0.2, 0.3):
    rs = [r for r in nf if r["phi"] == ph]; H = np.array([r["H"] for r in rs]) / 1e3; y = np.array([r["SAR_Fe"] for r in rs]); e = np.array([r["se_Fe"] for r in rs])
    ok = y > 2 * e
    a.errorbar(H[ok], y[ok], yerr=2 * e[ok], fmt="-" + mks[ph], color=cols[ph], ms=9, lw=2.4, capsize=4, label="Dispersed" if ph is None else f"Cluster, φ = {ph}")
a.axvline(4.5, color=GREY, ls="--", lw=1.6); a.text(4.9, 0.25, "tolerated\n4–5 kA m$^{-1}$", fontsize=15, color=BLACK)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(4, 27); a.set_ylim(0.2, 150)
a.set_xticks([5, 10, 20]); a.set_xticklabels(["5", "10", "20"]); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel("Field amplitude (kA m$^{-1}$)"); a.set_ylabel("SAR (W g$^{-1}$ Fe)"); a.legend(loc="upper left", fontsize=14.5); lab(a, "(a)", x=0.96, ha="right", y=0.14)
b = axs[1]; disp = {r["H"]: r["SAR_Fe"] for r in nf if r["phi"] is None}
for ph in (0.1, 0.2, 0.3):
    rs = [r for r in nf if r["phi"] == ph]; H = np.array([r["H"] for r in rs]); F = np.array([r["SAR_Fe"] / disp[r["H"]] for r in rs])
    b.plot(H / 1e3, np.clip(F, 0, None), "-" + mks[ph], color=cols[ph], ms=9, lw=2.4, label=f"φ = {ph}")
b.axhline(1, color=BLACK, lw=1.2); b.axvline(4.5, color=GREY, ls="--", lw=1.6)
b.set_xscale("log"); b.set_xlim(4, 27); b.set_xticks([5, 10, 20]); b.set_xticklabels(["5", "10", "20"]); b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); b.set_ylim(0, 3)
b.set_xlabel("Field amplitude (kA m$^{-1}$)"); b.set_ylabel("SAR$_{cluster}$ / SAR$_{dispersed}$"); b.legend(loc="upper left", fontsize=15); lab(b, "(b)", x=0.55, ha="center", y=0.95)
c = axs[2]; W = sorted({r["w"] for r in ps["rows"]})
for F, col, mk, nm in ((1.0, BLACK, "o", "Planned (fluid SAR)"), (0.52, BLUE, "s", "Cluster, φ = 0.1"), (0.17, GREEN, "^", "Cluster, φ = 0.2")):
    y = [37 + F * np.mean([r["dT50"] for r in ps["rows"] if r["w"] == w]) for w in W]
    c.plot(np.array(W) * 1e3, y, "-" + mk, color=col, ms=9, lw=2.4, label=nm)
c.axhline(40.8, color=RED, ls="--", lw=2.2); c.text(7.9, 41.15, "measured T$_{50}$", ha="right", color=RED, fontsize=15)
c.set_xlabel("Perfusion (10$^{-3}$ s$^{-1}$)"); c.set_ylabel("Prostate T$_{50}$ (°C)"); c.set_ylim(37, 48.5); c.set_xlim(0.5, 8.5)
c.legend(loc="upper right", fontsize=14.5); lab(c, "(c)", x=0.36, y=0.66)
d = axs[3]   # Johannsen 2005, probes 1-4, same field (4-5 kA/m), first vs sixth session
mx1, mx6 = [48.5, 43.0, 43.7, 43.6], [42.5, 42.3, 41.5, 40.7]; mn1, mn6 = [41.2, 40.3, 40.0, 41.1], [40.5, 39.8, 39.7, 39.4]
x = np.array(mx1 + mn1) - 37; y = np.array(mx6 + mn6) - 37; slope = float(np.sum(x * y) / np.sum(x * x))
d.plot(np.array(mx1) - 37, np.array(mx6) - 37, "o", color=RED, ms=11, label="Probe maximum"); d.plot(np.array(mn1) - 37, np.array(mn6) - 37, "s", color=BLUE, ms=11, label="Probe minimum")
xx = np.array([0, 13]); d.plot(xx, xx, "-", color=BLACK, lw=1.6, label="No change"); d.plot(xx, slope * xx, "--", color=GREY, lw=2.2, label=f"Fit, slope {slope:.2f}")
d.set_xlim(0, 13); d.set_ylim(0, 13); d.set_xlabel("ΔT, 1st session (°C)"); d.set_ylabel("ΔT, 6th session (°C)"); d.legend(loc="upper left", fontsize=14.5); lab(d, "(d)", x=0.96, ha="right", y=0.14)
fig.savefig("../figs/mnp_step2_clinic.png"); plt.close(fig); print("slope", round(slope, 3))
