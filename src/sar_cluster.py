import numpy as np, kmc
MS = 3.75e5      # A/m, Garcia-Acevedo 2025: 72.5 A m2/kg x 5170 kg/m3
def sizes(N, dmed, sig, rng): return dmed * np.exp(sig * rng.standard_normal(N))
def sar(kind, phi, dmed, K, H0, f, N=60, sig=0.12, gap=1.5, seeds=3, cycles=10, steps=300, Ms=MS, full=False):
    out, loops = [], []
    for sd in range(seeds):
        rng = np.random.default_rng(100 + sd); d = sizes(N, dmed, sig, rng)
        if kind == "dilute": pos = np.zeros((N, 3)); dip = False
        else: pos = kmc.cluster(N, dmed, gap, kind=kind, seed=sd, phi_target=phi); dip = True
        r = kmc.run(pos, d, K, Ms, H0, f, dip=dip, cycles=cycles, steps=steps, discard=3, seed=sd)
        out.append(r["SAR"] * 1e-3); loops.append(r["loop"])
    if full: return out, loops
    return float(np.mean(out)), float(np.std(out) / np.sqrt(seeds))

def hc_mr(loop):
    """coercive field (A/m) and remanence (A/m) from one cycle (H, M) by linear interpolation at zero crossings"""
    H, M = loop[:, 0], loop[:, 1]; hc, mr = [], []
    for i in range(len(H) - 1):
        if M[i] * M[i + 1] < 0: hc.append(abs(H[i] - M[i] * (H[i + 1] - H[i]) / (M[i + 1] - M[i])))
        if H[i] * H[i + 1] < 0: mr.append(abs(M[i] - H[i] * (M[i + 1] - M[i]) / (H[i + 1] - H[i])))
    return (np.mean(hc) if hc else 0.0), (np.mean(mr) if mr else 0.0)
