"""Fig. 1 (framework) and graphical abstract."""
from plotstyle import *
import json
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
def particles(ax, cx, cy, R, n, dmin, seed, col):
    rng = np.random.default_rng(seed); pts = []
    for _ in range(200000):
        if len(pts) == n: break
        p = rng.uniform(-R, R, 2)
        if np.hypot(*p) > R - dmin / 2: continue
        if all(np.hypot(*(p - q)) >= dmin for q in pts): pts.append(p)
    for p in pts:
        ax.add_patch(Circle((cx + p[0], cy + p[1]), dmin * 0.42, fc=col, ec=BLACK, lw=1.0))
        a = rng.uniform(0, 2 * np.pi); L = dmin * 0.38
        ax.annotate("", xy=(cx + p[0] + L * np.cos(a), cy + p[1] + L * np.sin(a)), xytext=(cx + p[0] - L * np.cos(a), cy + p[1] - L * np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, mutation_scale=8))
def box(ax, x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=col, lw=2.6))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=21, color=col, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.25, body, ha="center", va="center", fontsize=17, linespacing=1.35)
def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28, lw=2.6, color=BLACK))

# ---------------- Fig. 1
fig = plt.figure(figsize=(15, 9.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(0.6, 9.6); ax.axis("off")
ax.text(0.25, 9.3, "(a)", fontsize=26, va="top"); ax.text(0.25, 4.6, "(b)", fontsize=26, va="top")
for cx, R, n, dmin, lab_, sd, col in ((2.9, 1.55, 9, 0.42, "Dispersed (fluid)", 1, BLUE), (7.5, 1.05, 16, 0.38, "Soft aggregate, φ = 0.1", 2, ORANGE), (12.1, 0.80, 18, 0.30, "Dense aggregate, φ = 0.2", 3, RED)):
    particles(ax, cx, 7.35, R, n, dmin, sd, col); ax.text(cx, 5.45, lab_, ha="center", fontsize=19)
arrow(ax, 4.75, 7.35, 6.15, 7.35); arrow(ax, 8.85, 7.35, 10.95, 7.35)
ax.text(5.45, 7.75, "uptake", ha="center", fontsize=16); ax.text(9.9, 7.75, "compaction", ha="center", fontsize=16)
W, Hh, y = 2.6, 2.75, 1.0; xs = [0.35, 3.3, 6.25, 9.2, 12.15]
B = [("Particle", "Néel KMC\nwith dipolar\ncoupling\n→ SAR(φ, H, f)", BLUE), ("Field limit", "H·f ≤ 4.77×10$^{8}$\nA m$^{-1}$ s$^{-1}$\n50–200 kHz", PURPLE),
     ("Tissue", "Prostate,\n24 depots\nPennes equation\n→ T$_{90}$, T$_{50}$", ORANGE), ("Radiobiology", "Extended LQ\n28 × 2.5 Gy\n+ 6 sessions\n→ EQD gain", RED),
     ("Planning", "ANN surrogate\nband hold-out\n→ robust design", GREEN)]
for x, (t, b, c) in zip(xs, B): box(ax, x, y, W, Hh, t, b, c)
for i in range(4): arrow(ax, xs[i] + W + 0.03, y + Hh / 2, xs[i + 1] - 0.03, y + Hh / 2)
fig.savefig("../figs/mnp_fig1_framework.png"); plt.close(fig)

# ---------------- graphical abstract
T = json.load(open("../data/step3_therapy.json"))
g = lambda k: T[f"{k}|0.003"]["gain98"]
fig = plt.figure(figsize=(16, 6.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.4); ax.axis("off")
ax.text(4.0, 6.15, "Particles aggregate in tumour tissue", ha="center", va="top", fontsize=24)
for cx, R, n, dmin, sd, col, s in ((1.55, 1.15, 8, 0.42, 1, BLUE, "3.8"), (4.0, 0.85, 14, 0.33, 2, ORANGE, "2.0"), (6.45, 0.66, 16, 0.25, 3, RED, "0.8")):
    particles(ax, cx, 3.75, R, n, dmin, sd, col); ax.text(cx, 1.95, s, ha="center", fontsize=26, color=col)
ax.text(1.55, 1.35, "fluid", ha="center", fontsize=20); ax.text(4.0, 1.35, "φ = 0.1", ha="center", fontsize=20); ax.text(6.45, 1.35, "φ = 0.2", ha="center", fontsize=20)
ax.text(4.0, 0.55, "SAR (W g$^{-1}$ Fe) at 100 kHz, 4.77 kA m$^{-1}$", ha="center", fontsize=19)
a2 = fig.add_axes([0.575, 0.2, 0.405, 0.64])
labs = ["Fluid\nplan", "φ 0.1", "φ 0.2", "φ 0.1", "φ 0.2"]
E = json.load(open("../data/step4_extra.json"))["safe85"]["scen"]
vals = [g("Planned (dispersed)"), g("Clinical, φ = 0.1"), g("Clinical, φ = 0.2"), E["robust50|0.1"]["gain"], E["robust50|0.2"]["gain"]]
cols = [BLACK, BLUE, BLUE, GREEN, GREEN]
a2.bar(range(5), vals, 0.66, color=cols, ec=BLACK, lw=1.4)
for i, v in enumerate(vals): a2.text(i, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=21)
a2.set_xticks(range(5)); a2.set_xticklabels(labs, fontsize=19); a2.set_ylim(0, 8.2); a2.set_ylabel("EQD gain (Gy)", fontsize=23)
a2.tick_params(labelsize=19); a2.axvline(2.5, color=GREY, ls=":", lw=1.5)
a2.text(1.0, 7.9, "Clinical, 100 kHz", ha="center", va="top", fontsize=19); a2.text(3.5, 7.9, "Robust design", ha="center", va="top", fontsize=19)
a2.text(3.5, 7.2, "14 nm, 85 kHz", ha="center", va="top", fontsize=17)
fig.text(0.778, 0.955, "Radiotherapy benefit (28 × 2.5 Gy)", ha="center", va="top", fontsize=24)
fig.savefig("../figs/mnp_graphical_abstract.png"); plt.close(fig); print("ok", [round(v, 2) for v in vals])
