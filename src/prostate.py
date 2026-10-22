"""Steady-state Pennes heating of a prostate with intratumoural magnetic-fluid depots (FFT Green's function).
Homogeneous tissue (k, perfusion) around and inside the gland; Q = SAR_Fe(H) * c_Fe in each depot."""
import numpy as np
RHOC_B = 1060 * 3617.0
def grid(L=0.12, n=128):
    x = (np.arange(n) - n / 2 + 0.5) * (L / n); X, Y, Z = np.meshgrid(x, x, x, indexing="ij"); return x, X, Y, Z
def depots(rng, R_pro=0.0203, n_dep=24, r_dep=0.0053, r_ureth=0.004, dmin=0.0085, tries=20000):
    c = []
    for _ in range(tries):
        p = rng.uniform(-R_pro, R_pro, 3)
        if np.linalg.norm(p) > R_pro - 0.5 * r_dep: continue
        if np.hypot(p[0], p[1]) < r_ureth + r_dep: continue          # spare the urethra (z axis)
        if all(np.linalg.norm(p - q) >= dmin for q in c): c.append(p)
        if len(c) == n_dep: break
    return np.array(c)
def solve(Q, L, k=0.5, w=3e-3):
    n = Q.shape[0]; npad = 2 * n
    Qp = np.zeros((npad,) * 3); Qp[:n, :n, :n] = Q
    q = 2 * np.pi * np.fft.fftfreq(npad, d=L / n); QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij", sparse=True)
    T = np.fft.irfftn(np.fft.rfftn(Qp) / (k * (QX ** 2 + QY ** 2 + QZ[..., :npad // 2 + 1] ** 2) + w * RHOC_B), s=Qp.shape)
    return T[:n, :n, :n]
def run(sar_fe, H_kA=None, seed=0, w=3e-3, k=0.5, c_fe=100.7e3, L=0.12, n=128, R_pro=0.0203, r_dep=0.0053):
    """sar_fe: W per g Fe in the depots; c_fe: g Fe per m^3 of depot (100.7 mg/mL)"""
    rng = np.random.default_rng(seed); x, X, Y, Z = grid(L, n); C = depots(rng, R_pro=R_pro, r_dep=r_dep)
    dep = np.zeros(X.shape, bool)
    for p in C: dep |= (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2 <= r_dep ** 2
    Q = np.where(dep, sar_fe * c_fe, 0.0)
    T = 37.0 + solve(Q, L, k=k, w=w); pro = X ** 2 + Y ** 2 + Z ** 2 <= R_pro ** 2
    t = T[pro]
    return dict(T20=float(np.percentile(t, 80)), T50=float(np.percentile(t, 50)), T90=float(np.percentile(t, 10)),
                Tmax=float(t.max()), Tmin=float(t.min()), dep_frac=float(dep[pro].mean()), n_dep=len(C), T=T, pro=pro)
