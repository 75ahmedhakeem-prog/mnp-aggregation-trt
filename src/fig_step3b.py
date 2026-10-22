from plotstyle import *
import json
r = json.load(open("../data/step3_therapy.json"))
names = ["Planned (dispersed)", "Clinical, φ = 0.1", "Clinical, φ = 0.2", "Clinical 50 kHz, φ = 0.1", "Clinical 50 kHz, φ = 0.2", "Robust 50 kHz, φ = 0.1", "Robust 50 kHz, φ = 0.2"]
short = ["Fluid", "φ 0.1", "φ 0.2", "φ 0.1", "φ 0.2", "φ 0.1", "φ 0.2"]
cols = [BLACK, BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw=dict(wspace=0.28))
for a, key, ylab, lim in ((axs[0], "T90", "Prostate T$_{90}$ (°C)", (37, 45.5)), (axs[1], "gain98", "EQD gain at D$_{98}$ (Gy)", (0, 19))):
    x = np.arange(len(names))
    a.bar(x, [r[f"{n}|0.003"][key] - (37 if key == "T90" else 0) for n in names], 0.62, bottom=37 if key == "T90" else 0, color=cols, ec=BLACK, lw=1.2)
    a.plot(x, [r[f"{n}|0.002"][key] for n in names], "v", color=BLACK, ms=11, mfc="white", mew=2, label="Perfusion 2×10$^{-3}$ s$^{-1}$")
    a.plot(x, [r[f"{n}|0.005"][key] for n in names], "^", color=BLACK, ms=11, mfc=GREY, mew=1.5, label="Perfusion 5×10$^{-3}$ s$^{-1}$")
    a.set_xticks(x); a.set_xticklabels(short, fontsize=16, rotation=0); a.set_ylabel(ylab); a.set_ylim(*lim)
    for xx in (0.5, 2.5, 4.5): a.axvline(xx, color=GREY, ls=":", lw=1.2)
    yt = lim[1] - 0.06 * (lim[1] - lim[0])
    for xx, t in ((1.5, "100 kHz"), (3.5, "50 kHz"), (5.5, "Robust, 50 kHz")): a.text(xx, yt, t, ha="center", fontsize=15, fontweight="bold")
axs[0].axhline(41, color=RED, ls="--", lw=2); axs[0].text(2.0, 41.12, "41 °C", ha="center", va="bottom", color=RED, fontsize=15)
axs[1].legend(loc="upper center", bbox_to_anchor=(0.42, 0.86), fontsize=14); lab(axs[0], "(a)", y=0.83); lab(axs[1], "(b)", y=0.83)
fig.savefig("../figs/mnp_step3_therapy.png"); plt.close(fig); print("ok")
