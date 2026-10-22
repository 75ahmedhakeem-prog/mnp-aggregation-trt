from plotstyle import *
import json
from matplotlib.ticker import NullFormatter
E = json.load(open("../data/step4_extra.json")); P = E["progressive"]
fig, axs = plt.subplots(1, 3, figsize=(21, 6.6), gridspec_kw=dict(wspace=0.36))
a = axs[0]; s = np.arange(1, 7); m90 = E["T90_rise_per_SAR"]
a.plot(s, [37 + P["SAR_points"][0] * m90] * 6, "o-", color=BLACK, ms=10, label="Fluid, all sessions")
a.plot(s, P["T90_sessions_matched"], "s-", color=BLUE, ms=10, label=f"φ 0 → {P['phi6_matched']:.2f} (clinical drop)")
a.plot(s, P["T90_sessions_to02"], "^-", color=RED, ms=11, label="φ 0 → 0.20")
a.set_xlabel("Session"); a.set_ylabel("Prostate T$_{90}$ (°C)"); a.set_xticks(s); a.set_ylim(37.5, 43.5)
a.legend(loc="upper right", fontsize=15); a.text(0.04, 0.10, "(a)", transform=a.transAxes, fontsize=26, fontweight="bold")
R = E["required"]; F = np.array([5e4, 7e4, 1e5, 1.4e5, 2e5]) / 1e3
b = axs[1]
for part, ph, mk, col, lab_ in (("clinical", 0.1, "s", BLUE, "Clinical, φ 0.1"), ("clinical", 0.2, "^", BLUE, "Clinical, φ 0.2"),
                                ("robust", 0.1, "s", GREEN, "Robust, φ 0.1"), ("robust", 0.2, "^", GREEN, "Robust, φ 0.2")):
    v = [next(r["x_req"] for r in R if r["particle"] == part and r["phi"] == ph and r["f"] / 1e3 == f) for f in F]
    fin = [x if np.isfinite(x) else np.nan for x in v]
    b.plot(F, fin, mk + ("-" if ph == 0.1 else "--"), color=col, ms=11, mfc=col if ph == 0.1 else "white", mew=2, label=lab_)
    for f, x in zip(F, v):
        if not np.isfinite(x): b.plot(f, 2.3, mk, color=col, ms=11, mfc="white", mew=2)
b.axhline(1.0, color=BLACK, lw=1.6); b.text(205, 0.93, "safety limit", ha="right", va="top", fontsize=15)
b.text(178, 2.3, "> 2×", ha="right", va="center", fontsize=15)
b.set_xscale("log"); b.set_xticks([50, 100, 200]); b.set_xticklabels(["50", "100", "200"]); b.xaxis.set_minor_formatter(NullFormatter())
b.set_xlabel("Frequency (kHz)"); b.set_ylabel("(H·f)$_{req}$ / limit for T$_{90}$ = 41 °C"); b.set_ylim(0.3, 2.5)
b.legend(loc="upper left", fontsize=14, ncol=1, bbox_to_anchor=(0.0, 0.88)); b.text(0.04, 0.97, "(b)", transform=b.transAxes, fontsize=26, fontweight="bold", va="top")
c = axs[2]
for part, ph, mk, col in (("clinical", 0.1, "s", BLUE), ("clinical", 0.2, "^", BLUE), ("robust", 0.1, "s", GREEN), ("robust", 0.2, "^", GREEN)):
    g = [next(r["gain_at_limit"] for r in R if r["particle"] == part and r["phi"] == ph and r["f"] / 1e3 == f) for f in F]
    c.plot(F, g, mk + ("-" if ph == 0.1 else "--"), color=col, ms=11, mfc=col if ph == 0.1 else "white", mew=2)
fm = E["safe85"]["f_min"] / 1e3
c.axvline(fm, color=RED, ls=":", lw=2.2); c.text(fm * 0.97, 0.93, "nerve\nstimulation\nrisk", transform=c.get_xaxis_transform(), ha="right", va="top", fontsize=15, color=RED)
c.set_xscale("log"); c.set_xticks([50, 100, 200]); c.set_xticklabels(["50", "100", "200"]); c.xaxis.set_minor_formatter(NullFormatter())
c.set_xlabel("Frequency at the limit (kHz)"); c.set_ylabel("EQD gain at D$_{98}$ (Gy)"); c.set_yscale("log"); c.set_ylim(0.2, 60)
c.text(0.96, 0.97, "(c)", transform=c.transAxes, fontsize=26, fontweight="bold", va="top", ha="right")
fig.savefig("../figs/mnp_step4_field.png"); plt.close(fig); print("ok")
