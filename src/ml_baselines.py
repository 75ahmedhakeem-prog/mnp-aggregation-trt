"""Smooth-function baselines for the SAR surrogate: polynomial response surfaces (degree 2-5), additive + pairwise-interaction
cubic splines, local multiquadric RBF interpolation and Gaussian-process regression (ARD Matern 5/2 + white noise, because the
KMC targets are noisy). The random split (rng 0, 80/20) and the six interior band hold-outs of ml_sar.py are replayed exactly,
so every baseline is trained and tested on the same samples as the ANN ensemble and the GBT of data/ml_sar.json."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.interpolate import RBFInterpolator
import ml_sar as M

def poly(deg): return make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=1e-6, solver="cholesky"))
def spline():
    # cubic B-splines with 5 quantile knots per input, then all pairwise products of the basis functions (additive + 2-way);
    # the ~860 tensor features exceed what 960 noisy samples support, so the ridge penalty is chosen by internal
    # leave-one-out CV on the training set only (a fixed 1e-3 overfits: random-split R2 0.63)
    return make_pipeline(SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear"),
                         PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
class RBF:
    def fit(self, X, y):
        self.s = StandardScaler().fit(X); self.m = y.mean()
        self.f = RBFInterpolator(self.s.transform(X), y - self.m, kernel="multiquadric", epsilon=1.0, neighbors=64, smoothing=1e-6, degree=0); return self
    def predict(self, X): return self.f(self.s.transform(X)) + self.m
class GP:
    """ARD Matern 5/2 + WhiteKernel; with <= 1200 samples the hyperparameters are optimised on the full training set
    (3 restarts); prediction in batches of 1000 points."""
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X, y):
        X = np.asarray(X, float); self.sx = StandardScaler().fit(X); self.my, self.sy = y.mean(), y.std()
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1.0))
        self.g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=3, random_state=self.seed).fit(self.sx.transform(X), (y - self.my) / self.sy)
        return self
    def predict(self, X):
        Z = self.sx.transform(np.asarray(X, float))
        return np.concatenate([self.g.predict(Z[i:i + 1000]) for i in range(0, len(Z), 1000)]) * self.sy + self.my
MODELS = {"Poly2": lambda: poly(2), "Poly3": lambda: poly(3), "Poly4": lambda: poly(4), "Poly5": lambda: poly(5),
          "Spline": spline, "RBF": lambda: RBF(), "GP": lambda: GP()}

def fit_eval(nm, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); m = MODELS[nm]().fit(Xtr, ytr); t1 = time.perf_counter(); p = m.predict(Xte); t2 = time.perf_counter()
    out = dict(M.met(yte, p), t_fit=t1 - t0)
    if nm == "GP": out["kernel"] = str(m.g.kernel_)
    return out

def bands():
    """the six interior band hold-outs of ml_sar.py (same bounds)"""
    for j, nm_v in enumerate(M.NAMES):
        if nm_v == "phi": lo, hi = 0.13, 0.20
        else: lo, hi = np.quantile(M.X[:, j], [0.40, 0.60])
        yield nm_v, float(lo), float(hi), (M.X[:, j] >= lo) & (M.X[:, j] <= hi)

if __name__ == "__main__":
    X, y = M.X, M.y; res = {"n": len(y), "random": {}, "band": []}
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); n = int(0.8 * len(y)); tr, te = idx[:n], idx[n:]
    for nm in MODELS: res["random"][nm] = fit_eval(nm, X[tr], y[tr], X[te], y[te])
    print("random", {k: round(v["R2"], 4) for k, v in res["random"].items()}, flush=True)
    for var, lo, hi, mask in bands():
        row = dict(variable=var, lo=lo, hi=hi, n_test=int(mask.sum()))
        for nm in MODELS:
            r = fit_eval(nm, X[~mask], y[~mask], X[mask], y[mask]); row[f"{nm}_R2"] = r["R2"]; row[f"{nm}_RMSE"] = r["RMSE"]
        res["band"].append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    ref = json.load(open("../data/ml_sar.json")); res["noise_ceiling_R2"] = ref["noise_ceiling_R2"]
    # summary: mean and minimum band R2 for every model, ANN and GBT taken from ml_sar.json (same split and bands)
    allm = ["ANN", "GBT", *MODELS]; summ = {}
    for nm in allm:
        rr = ref["random"][nm]["R2"] if nm in ref["random"] else res["random"][nm]["R2"]
        bb = [b[f"{nm}_R2"] for b in (ref["band"] if nm in ("ANN", "GBT") else res["band"])]
        summ[nm] = dict(random_R2=rr, band_mean_R2=float(np.mean(bb)), band_min_R2=float(np.min(bb)))
    res["summary"] = summ
    json.dump(res, open("../data/ml_baselines.json", "w"), indent=1)
    for k, v in summ.items(): print(f"{k:7s} random {v['random_R2']:.3f}  band mean {v['band_mean_R2']:.3f}  min {v['band_min_R2']:.3f}")
