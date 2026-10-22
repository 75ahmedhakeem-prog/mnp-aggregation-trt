"""Cache prostate temperature-rise maps per unit SAR (W/g Fe) for eight depot layouts and several perfusion / conductivity values."""
import os, numpy as np, prostate as P
FN = "../data/maps_cache.npz"
CASES = [(2e-3, 0.5), (3e-3, 0.5), (5e-3, 0.5), (3e-3, 0.4), (3e-3, 0.6), (1e-3, 0.5), (4e-3, 0.5), (8e-3, 0.5)]
def key(w, k): return f"w{w:.0e}_k{k}"
def load():
    if os.path.exists(FN):
        z = np.load(FN); return {k: z[k] for k in z.files}
    out = {}
    for w, k in CASES:
        out[key(w, k)] = np.array([(lambda r: r["T"][r["pro"]] - 37.0)(P.run(1.0, seed=sd, w=w, k=k)) for sd in range(8)], dtype=np.float32)
        print("map", w, k, flush=True)
    np.savez_compressed(FN, **out); return out
if __name__ == "__main__": m = load(); print({k: v.shape for k, v in m.items()})
