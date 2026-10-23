from plotstyle import *
import json
CI = json.load(open("../data/clin_inverse.json")); G = json.load(open("../data/gain_phi.json"))
fig, axs = plt.subplots(1, 3, figsize=(21, 6.6), gridspec_kw=dict(wspace=0.36))
a = axs[0]; S = np.array(CI["S"]); W = np.array(CI["W"]) * 1e3; chi = np.array(CI["chi"]); dc = chi - chi.min()
cs = a.contour(S, W, dc, levels=[2.3, 6.2, 11.8], colors=[BLACK, BLUE, GREY], linewidths=[2.4, 2.0, 1.6], linestyles=["-", "--", ":"])
a.clabel(cs, fmt={2.3: "68%", 6.2: "95%", 11.8: "99.7%"}, fontsize=14, inline=True)
a.plot(CI["best"]["SAR"], CI["best"]["w"] * 1e3, "*", color=RED, ms=22, mec=BLACK)
fl = float(np.mean([CI["sar_table"]["4500"]["0.0"]]))
a.axvline(fl, color=RED, ls="-.", lw=1.8); a.text(fl - 0.08, 7.6, "fluid SAR", rotation=90, ha="right", va="top", fontsize=15, color=RED)
a.set_xlim(0.2, 5.0); a.set_ylim(1, 8); a.set_xlabel("Effective SAR (W g$^{-1}$ Fe)"); a.set_ylabel("Perfusion (10$^{-3}$ s$^{-1}$)")
a.text(0.96, 0.96, "(a)", transform=a.transAxes, fontsize=26, fontweight="bold", va="top", ha="right")
b = axs[1]; T = CI["sar_table"]
for H, mk, col in (("4000", "o", GREEN), ("4500", "s", BLUE), ("5000", "^", ORANGE)):
    ph = sorted(T[H], key=float); b.plot([float(p) for p in ph], [max(T[H][p], 0.0) for p in ph], mk + "-", color=col, ms=10, label=f"{int(H) / 1000:.1f} kA m$^{{-1}}$")
for v in CI["SAR_range"]: b.axhline(v, color=BLACK, ls="--", lw=1.6)
b.text(0.29, CI["SAR_range"][1] + 0.08, "clinical range", ha="right", va="bottom", fontsize=15)
b.set_xlabel("Packing fraction, φ"); b.set_ylabel("SAR (W g$^{-1}$ Fe)"); b.set_xlim(-0.01, 0.31); b.set_ylim(0, 5)
b.legend(loc="upper right", fontsize=15); b.text(0.04, 0.10, "(b)", transform=b.transAxes, fontsize=26, fontweight="bold")
c = axs[2]
for part, mk, col, lab_ in (("clinical", "s", BLUE, "Clinical, 100 kHz"), ("robust", "o", GREEN, "Robust, 85 kHz")):
    d = G[part]; c.plot(d["phi"], d["gain"], mk + "-", color=col, ms=10, label=lab_)
c.set_xlabel("Packing fraction, φ"); c.set_ylabel("EQD gain at D$_{98}$ (Gy)"); c.set_xlim(-0.01, 0.31); c.set_ylim(0, 9)
c.annotate("61 Gy\n(T$_{90}$ > 45 °C)", xy=(0.012, 8.9), xytext=(0.075, 6.6), fontsize=14, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
c.legend(loc="upper right", fontsize=15); c.text(0.04, 0.10, "(c)", transform=c.transAxes, fontsize=26, fontweight="bold")
fig.savefig("../figs/mnp_step5_phi.png"); plt.close(fig); print("ok")
