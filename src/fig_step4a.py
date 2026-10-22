from plotstyle import *
import json
U = json.load(open("../data/uq_therapy.json"))
fig = plt.figure(figsize=(16, 12.5)); gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.42)
LAB = ["Fluid", "φ 0.1", "φ 0.2", "φ 0.1", "φ 0.2", "φ 0.1", "φ 0.2"]; KEYS = ["fluid", "c01", "c02", "c50_01", "c50_02", "r01", "r02"]
COL = [BLACK, BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
# (a) gains with 95% intervals
a = fig.add_subplot(gs[0, 0]); ci = U["ci"]; base = U["sens"]["base"]
x = np.arange(7); g = [ci[k]["mean"] for k in KEYS]
a.bar(x, g, 0.66, color=COL, ec=BLACK, lw=1.2)
a.errorbar(x, g, yerr=[[g[i] - ci[k]["lo"] for i, k in enumerate(KEYS)], [ci[k]["hi"] - g[i] for i, k in enumerate(KEYS)]], fmt="none", ecolor=BLACK, capsize=6, lw=1.8)
a.set_xticks(x); a.set_xticklabels(LAB, fontsize=17, rotation=0); a.set_ylabel("EQD gain at D$_{98}$ (Gy)"); a.set_ylim(0, 30)
for xx in (0.5, 2.5, 4.5): a.axvline(xx, color=GREY, ls=":", lw=1.2)
a.text(1.5, 29, "100 kHz", ha="center", va="top", fontsize=17); a.text(3.5, 29, "50 kHz", ha="center", va="top", fontsize=17); a.text(5.5, 29, "Robust", ha="center", va="top", fontsize=17)
a.text(0.04, 0.80, "(a)", transform=a.transAxes, fontsize=26, fontweight="bold")
# (b) loss fraction under one-at-a-time changes
b = fig.add_subplot(gs[0, 1]); S = U["sens"]
V = [k for k in S if k not in ("beta(T) as SiHa",)]
NM = {"base": "Base case", "perfusion 2e-3": "ω$_b$ 2×10$^{-3}$ s$^{-1}$", "perfusion 5e-3": "ω$_b$ 5×10$^{-3}$ s$^{-1}$", "k 0.4": "k 0.4 W m$^{-1}$ K$^{-1}$",
      "k 0.6": "k 0.6 W m$^{-1}$ K$^{-1}$", "alpha/beta 3 Gy": "α/β 3 Gy", "alpha37 0.10": "α$_{37}$ 0.10 Gy$^{-1}$", "alpha37 0.30": "α$_{37}$ 0.30 Gy$^{-1}$",
      "HeLa thermal ratio": "HeLa α ratio", "interval 1 h": "Interval 1 h", "no direct kill": "No direct kill", "45 min sessions": "45 min sessions"}
y = np.arange(len(V))[::-1]
b.plot([100 * S[k]["loss01"] for k in V], y, "s", color=BLUE, ms=11, label="φ = 0.1")
b.plot([100 * S[k]["loss02"] for k in V], y, "^", color=RED, ms=12, label="φ = 0.2")
b.axvline(100 * S["base"]["loss01"], color=BLUE, ls="--", lw=1.4); b.axvline(100 * S["base"]["loss02"], color=RED, ls="--", lw=1.4)
b.set_yticks(y); b.set_yticklabels([NM[k] for k in V], fontsize=15); b.set_xlim(0, 100); b.set_xlabel("Planned gain lost (%)")
b.legend(loc="lower left", fontsize=16); b.text(0.04, 0.97, "(b)", transform=b.transAxes, fontsize=26, fontweight="bold", va="top")
b.tick_params(axis="y", which="both", right=False, left=True)
# (c) isoeffective radiotherapy dose from TCP
c = fig.add_subplot(gs[1, 0]); T = U["tcp"]
mid = [T["0.15"]["scen"][k]["iso_dose"] - 70 for k in KEYS]
lo = [min(T[s]["scen"][k]["iso_dose"] for s in T) - 70 for k in KEYS]; hi = [max(T[s]["scen"][k]["iso_dose"] for s in T) - 70 for k in KEYS]
c.bar(x, mid, 0.66, color=COL, ec=BLACK, lw=1.2)
c.errorbar(x, mid, yerr=[np.array(mid) - lo, np.array(hi) - mid], fmt="none", ecolor=BLACK, capsize=6, lw=1.8)
c.set_xticks(x); c.set_xticklabels(LAB, fontsize=17); c.set_ylabel("Isoeffective gain (Gy)"); c.set_ylim(0, 25)
for xx in (0.5, 2.5, 4.5): c.axvline(xx, color=GREY, ls=":", lw=1.2)
c.text(0.04, 0.80, "(c)", transform=c.transAxes, fontsize=26, fontweight="bold")
c.text(1.5, 24.2, "100 kHz", ha="center", va="top", fontsize=17); c.text(3.5, 24.2, "50 kHz", ha="center", va="top", fontsize=17); c.text(5.5, 24.2, "Robust", ha="center", va="top", fontsize=17)
# (d) TCP at 70 Gy
d = fig.add_subplot(gs[1, 1]); tc = [100 * T["0.15"]["scen"][k]["TCP"] for k in KEYS]
d.bar(x, tc, 0.66, color=COL, ec=BLACK, lw=1.2); d.axhline(86, color=BLACK, ls="--", lw=1.6)
d.text(6.6, 87.3, "RT alone", ha="right", fontsize=16)
d.set_xticks(x); d.set_xticklabels(LAB, fontsize=17); d.set_ylabel("TCP at 70 Gy (%)"); d.set_ylim(80, 103.5)
for xx in (0.5, 2.5, 4.5): d.axvline(xx, color=GREY, ls=":", lw=1.2)
d.text(1.5, 102.4, "100 kHz", ha="center", va="top", fontsize=17); d.text(3.5, 102.4, "50 kHz", ha="center", va="top", fontsize=17); d.text(5.5, 102.4, "Robust", ha="center", va="top", fontsize=17)
d.text(0.04, 0.80, "(d)", transform=d.transAxes, fontsize=26, fontweight="bold", va="top")
fig.savefig("../figs/mnp_step4_uq.png"); plt.close(fig); print("ok")
