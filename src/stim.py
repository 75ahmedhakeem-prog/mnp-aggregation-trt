"""Peripheral-nerve magnetostimulation index at the H*f limit (Saritas et al. 2013 full-body estimate):
threshold B_pp(f) = 14.3 mT * (1 + 1/(2 f tau_c)), tau_c = 289 us; index = 2 mu0 H / B_pp."""
import numpy as np
MU0, LIMIT, BMIN, TAUC = 4e-7 * np.pi, 4.77e8, 14.3e-3, 289e-6
def bth(f): return BMIN * (1 + 1 / (2 * f * TAUC))
def index(f, H=None):
    H = LIMIT / f if H is None else H; return 2 * MU0 * H / bth(f)
def fmin():
    from scipy.optimize import brentq
    return brentq(lambda f: index(f) - 1.0, 2e4, 5e5)
if __name__ == "__main__":
    for f in (5e4, 7e4, 1e5, 1.4e5, 2e5): print(int(f / 1e3), "kHz H", round(LIMIT / f), "index", round(index(f), 3))
    print("f_min", round(fmin()))
