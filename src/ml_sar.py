"""Surrogate of the aggregate KMC model: ANN vs gradient-boosted trees; random split and interior band hold-out."""
import json, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error
D = json.load(open("../data/ml_dataset.json"))
def feats(r): return [r["d"], np.log10(r["K"]), r["Ms"] / 1e5, np.log10(r["f"]), np.log10(r["H"]), r["phi"]]
X = np.array([feats(r) for r in D]); SHIFT = 0.5   # W/g Fe: log10(SAR + 0.5) suppresses the Monte Carlo noise of near-zero losses
y = np.array([np.log10(max(r["SAR_Fe"], 0.0) + SHIFT) for r in D])
NAMES = ["d", "logK", "Ms", "logf", "logH", "phi"]
def ann(seed=0):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh", alpha=1e-3,
        learning_rate_init=2e-3, batch_size=64, max_iter=3000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=80, random_state=seed)), transformer=StandardScaler())
class Ens:
    """mean of five ANNs that differ only in initialisation"""
    def fit(self, X, y): self.m = [ann(s).fit(X, y) for s in range(5)]; return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.m], 0)
def gbt(): return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
def met(yt, yp): return dict(R2=float(r2_score(yt, yp)), RMSE=float(np.sqrt(mean_squared_error(yt, yp))))
if __name__ == "__main__":
    res = {"n": len(D), "random": {}, "band": []}
    rng = np.random.default_rng(0); idx = rng.permutation(len(D)); n = int(0.8 * len(D)); tr, te = idx[:n], idx[n:]
    for nm, mk in (("ANN", Ens), ("GBT", gbt)):
        m = mk().fit(X[tr], y[tr]); res["random"][nm] = met(y[te], m.predict(X[te]))
    for j, nm_v in enumerate(NAMES):
        if nm_v == "phi": lo, hi = 0.13, 0.20
        else: q = np.quantile(X[:, j], [0.40, 0.60]); lo, hi = q
        mask = (X[:, j] >= lo) & (X[:, j] <= hi)
        row = dict(variable=nm_v, lo=float(lo), hi=float(hi), n_test=int(mask.sum()))
        for nm, mk in (("ANN", Ens), ("GBT", gbt)):
            m = mk().fit(X[~mask], y[~mask]); mm = met(y[mask], m.predict(X[mask])); row[nm + "_R2"] = mm["R2"]; row[nm + "_RMSE"] = mm["RMSE"]
        res["band"].append(row); print(row, flush=True)
    nv = np.mean([(r["se_Fe"] / (max(r["SAR_Fe"], 0) + SHIFT) / np.log(10)) ** 2 for r in D]); noise = 1 - nv / y.var(); res["noise_ceiling_R2"] = float(noise)
    json.dump(res, open("../data/ml_sar.json", "w"), indent=1); print(res["random"], "noise ceiling R2", round(noise, 3))
