from plotstyle import *
import json
r = json.load(open("../data/design_map.json"))
DS = [10, 12, 14, 16, 18]; KS = [0.8e4, 1.2e4, 1.6e4, 2.0e4, 2.6e4, 3.4e4]
def grid(f, ph):
    return np.array([[next(x["SAR_Fe"] for x in r if x["f"] == f and x["phi"] == ph and x["d"] == d and x["K"] == K) for d in DS] for K in KS])
fig, axs = plt.subplots(2, 2, figsize=(14.5, 11), gridspec_kw=dict(wspace=0.62, hspace=0.42)); axs = axs.ravel()
panels = [(1e5, None, "Dispersed, 100 kHz", 0), (1e5, 0.2, "Cluster φ = 0.2, 100 kHz", 1), (5e4, 0.2, "Cluster φ = 0.2, 50 kHz", 2)]
for f, ph, title, i in panels:
    a = axs[i]; G = np.clip(grid(f, ph), 0, None); vmax = 20 if ph is None else 5
    im = a.imshow(G, origin="lower", cmap="viridis", vmin=0, vmax=vmax, aspect="auto", extent=[-0.5, 4.5, -0.5, 5.5])
    for yi in range(len(KS)):
        for xi in range(len(DS)):
            v = G[yi, xi]; a.text(xi, yi, f"{v:.1f}", ha="center", va="center", fontsize=14.5, fontweight="bold", color="black" if v > 0.6 * vmax else "white")
    a.set_xticks(range(5)); a.set_xticklabels(DS); a.set_yticks(range(6)); a.set_yticklabels([f"{k/1e4:.1f}" for k in KS])
    a.set_xlabel("Core diameter (nm)"); a.set_ylabel("K (10$^4$ J m$^{-3}$)"); a.set_title("(" + "abc"[i] + ") " + title, fontsize=19, fontweight="bold")
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03); cb.set_label("SAR (W g$^{-1}$ Fe)", fontsize=18)
    a.tick_params(top=False, right=False)
d = axs[3]; FS = [5e4, 1e5, 2e5]
for ph, col, mk, nm in ((None, BLACK, "o", "Dispersed"), (0.1, BLUE, "s", "φ = 0.1"), (0.2, GREEN, "^", "φ = 0.2")):
    best = [max(x["SAR_Fe"] for x in r if x["f"] == f and x["phi"] == ph) for f in FS]
    d.plot(np.array(FS) / 1e3, best, "-" + mk, color=col, ms=10, lw=2.4, label=nm)
d.set_xscale("log"); d.set_yscale("log"); d.set_xticks([50, 100, 200]); d.set_xticklabels(["50", "100", "200"]); d.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
d.set_xlabel("Frequency at H·f = 4.77×10$^8$ (kHz)"); d.set_ylabel("Best SAR (W g$^{-1}$ Fe)"); d.set_ylim(0.8, 60)
d.legend(loc="upper right", fontsize=15); lab(d, "(d)", y=0.12)
fig.savefig("../figs/mnp_step3_design.png"); plt.close(fig); print("ok")
