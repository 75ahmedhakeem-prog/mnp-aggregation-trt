"""Kinetic Monte Carlo (two-state Neel) model of AC hysteresis losses in interacting magnetic nanoparticle clusters.
Each particle: uniaxial anisotropy K, volume V, easy axis n, moment Ms V s n (s = +/-1).
Barriers and minima from the exact Stoner-Wohlfarth energy evaluated on an angular grid (moment tilt included);
the two-state master equation is advanced with an exponential integrator with first-order hold (keeps the relaxation lag);
dipolar fields are recomputed every step from the current states. SAR = f * mu0 * loop area / rho."""
import numpy as np
MU0, KB = 4e-7 * np.pi, 1.380649e-23

def hsw(psi):
    c, s = np.abs(np.cos(psi)), np.abs(np.sin(psi))
    return 1.0 / (c ** (2 / 3) + s ** (2 / 3)) ** 1.5

def dip_tensor(pos):
    """3N x 3N matrix D so that H_dip = D @ m (A/m), m in A m^2, pos in m"""
    N = len(pos); D = np.zeros((N, 3, N, 3))
    r = pos[:, None, :] - pos[None, :, :]; d = np.linalg.norm(r, axis=2); np.fill_diagonal(d, np.inf)
    u = r / d[..., None]
    for a in range(3):
        for b in range(3):
            D[:, a, :, b] = (3 * u[..., a] * u[..., b] - (a == b)) / (4 * np.pi * d ** 3)
    return D.reshape(3 * N, 3 * N)

def run(pos, d_nm, K, Ms, H0, f, T=310.0, f0=1e9, cycles=12, steps=800, discard=4, seed=0, n=None, rho=5170.0, dip=True):
    rng = np.random.default_rng(seed); N = len(d_nm)
    V = np.pi / 6 * (np.asarray(d_nm) * 1e-9) ** 3
    if n is None:
        n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    HK = 2 * K / (MU0 * Ms); E0 = K * V / (KB * T); mag = Ms * V
    D = dip_tensor(pos) if (dip and N > 1) else None
    perp0 = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])); perp0 /= np.linalg.norm(perp0, axis=1)[:, None]
    s = rng.choice([-1.0, 1.0], N); u = s[:, None] * n; dt = 1.0 / (f * steps); W = 0.0; Mz_prev = None; H_prev = None
    zhat = np.array([0, 0, 1.0]); Mtot = []
    for k in range(cycles * steps):
        t = k * dt; Ha = H0 * np.cos(2 * np.pi * f * t); Ha1 = H0 * np.cos(2 * np.pi * f * (t + dt))
        Hd = (D @ (mag[:, None] * u).ravel()).reshape(N, 3) if D is not None else 0.0
        peqs, Gs = [], []
        for HH in (Ha, Ha1):
            Hv = HH * zhat[None, :] + Hd
            Hm = np.linalg.norm(Hv, axis=1) + 1e-30; hpar = np.einsum("ij,ij->i", Hv, n)
            Eplus, Eminus, tp, tm = barriers(Hm / HK, np.arccos(np.clip(hpar / Hm, -1, 1)), E0, want_angles=True)
            hperp = Hv - hpar[:, None] * n; hn = np.linalg.norm(hperp, axis=1)
            e2 = np.where(hn[:, None] > 1e-12, hperp / np.maximum(hn, 1e-30)[:, None], perp0)
            rp, rm = f0 * np.exp(-Eplus), f0 * np.exp(-Eminus); peqs.append(rm / (rp + rm)); Gs.append(rp + rm)
        G = 0.5 * (Gs[0] + Gs[1]); x = G * dt; ex = np.exp(-x)
        pnow = (s > 0).astype(float)
        # exponential integrator with linearly varying equilibrium (first-order hold): preserves the relaxation lag
        pnext = peqs[1] + (pnow - peqs[0]) * ex - (peqs[1] - peqs[0]) * (1 - ex) / np.maximum(x, 1e-300) * (x > 1e-12) \
                - (peqs[1] - peqs[0]) * (x <= 1e-12)
        Ha = Ha1
        s = np.where(rng.random(N) < pnext, 1.0, -1.0)
        up_dir = np.cos(tp)[:, None] * n + np.sin(tp)[:, None] * e2; dn_dir = np.cos(tm)[:, None] * n + np.sin(tm)[:, None] * e2
        u = np.where(s[:, None] > 0, up_dir, dn_dir)
        Mz = np.sum(mag * (pnext * up_dir[:, 2] + (1 - pnext) * dn_dir[:, 2])) / V.sum()   # conditional expectation, A/m
        if k >= discard * steps and Mz_prev is not None:
            W += MU0 * 0.5 * (Mz + Mz_prev) * (Ha - H_prev)
        Mz_prev, H_prev = Mz, Ha
        if k >= (cycles - 1) * steps: Mtot.append((Ha, Mz))
    loss = -W / (cycles - discard)                      # J/m^3 per cycle (loop area, sign: dissipated)
    return dict(SAR=f * loss / rho, loss_per_cycle=loss, loop=np.array(Mtot))

def lrt_sar(d_nm, K, Ms, H0, f, T=310.0, f0=1e9, rho=5170.0):
    """linear response (Rosensweig) for immobilised two-state particles with random easy axes"""
    V = np.pi / 6 * (d_nm * 1e-9) ** 3; sig = K * V / (KB * T)
    tau = np.exp(sig) / (2 * f0); chi0 = MU0 * Ms ** 2 * V / (3 * KB * T)
    w = 2 * np.pi * f; chi2 = chi0 * w * tau / (1 + (w * tau) ** 2)
    return MU0 * np.pi * f * chi2 * H0 ** 2 / rho

TH = np.linspace(-np.pi, np.pi, 181)[:-1]
DTH = TH[1] - TH[0]
def _refine(e, i):
    """parabolic refinement of a gridded minimum (periodic grid)"""
    G = e.shape[1]; r = np.arange(len(i)); a, b, c = e[r, (i - 1) % G], e[r, i], e[r, (i + 1) % G]
    den = a - 2 * b + c; off = np.where(den > 1e-15, 0.5 * (a - c) / np.maximum(den, 1e-15), 0.0)
    return TH[i] + np.clip(off, -0.5, 0.5) * DTH
def barriers(h, psi, E0, want_angles=False):
    """exact (gridded) Stoner-Wohlfarth barriers in units of kT for leaving the + (theta~0) and - (theta~pi) states.
    e(theta) = sin^2(theta) - 2 h cos(theta - psi) (units of KV), psi = angle between field and +n."""
    e = np.sin(TH)[None, :] ** 2 - 2 * h[:, None] * np.cos(TH[None, :] - psi[:, None])
    up = np.abs(TH) < np.pi / 2
    ep = e[:, up].min(1); em = e[:, ~up].min(1)
    mA = e[:, (TH > 0)].max(1); mB = e[:, (TH < 0)].max(1); saddle = np.minimum(mA, mB)
    if want_angles:
        tp = _refine(e, np.flatnonzero(up)[e[:, up].argmin(1)]); tm = _refine(e, np.flatnonzero(~up)[e[:, ~up].argmin(1)])
        return E0 * np.clip(saddle - ep, 0, None), E0 * np.clip(saddle - em, 0, None), tp, tm
    return E0 * np.clip(saddle - ep, 0, None), E0 * np.clip(saddle - em, 0, None)

# ---------- cluster geometries
def cluster(N, d_core_nm, gap_nm, kind="loose", seed=0, phi_target=0.2):
    """'loose'/'compact': N spheres (diameter d_core+gap) packed at volume fraction phi_target inside a sphere,
    generated by random placement followed by iterative overlap removal; 'chain': straight chain."""
    rng = np.random.default_rng(seed); dmin = (d_core_nm + gap_nm) * 1e-9
    if kind == "chain":
        return np.array([[0, 0, i * dmin] for i in range(N)]) @ _randrot(rng).T
    R = 0.5 * dmin * (N / phi_target) ** (1 / 3)
    p = rng.normal(size=(N, 3)); p *= (R - dmin / 2) * rng.random(N)[:, None] ** (1 / 3) / np.linalg.norm(p, axis=1)[:, None]
    for it in range(4000):
        r = p[:, None, :] - p[None, :, :]; d = np.linalg.norm(r, axis=2); np.fill_diagonal(d, np.inf)
        ov = np.clip(dmin - d, 0, None)
        if ov.max() < 1e-4 * dmin: break
        p += 0.5 * np.sum(ov[..., None] * r / d[..., None], axis=1)
        rr = np.linalg.norm(p, axis=1); out = rr > R - dmin / 2
        p[out] *= ((R - dmin / 2) / rr[out])[:, None]
    else:
        raise RuntimeError("packing did not converge")
    return p

def _randrot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
