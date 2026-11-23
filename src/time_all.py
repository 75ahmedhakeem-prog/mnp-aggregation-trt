"""Cost evidence for the surrogate, on one core of an otherwise idle machine (OMP/BLAS threads = 1):
 - one KMC SAR evaluation as used for the dataset (4 aggregates x 8 cycles), median over 12 dataset parameter sets;
 - the implied cost of the 1200-run dataset and of an exhaustive KMC search of the 6561-point (d, K) design grid;
 - fit time of every surrogate on the random 80/20 split of ml_sar.py (960 samples), median of three repetitions;
 - prediction time of every surrogate for the 6561-point design grid of ml_design.py (one state)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import json, time, platform, warnings; warnings.filterwarnings("ignore")
import numpy as np
import sar_cluster as S, gen_ml as G, ml_sar as M, ml_baselines as B
out = {"kmc": [], "models": {}}
for i in range(12):
    p = G.sample(i); t0 = time.perf_counter()
    S.sar("dilute" if p["phi"] == 0 else "loose", p["phi"] or None, p["d"], p["K"], p["H"], p["f"], Ms=p["Ms"], sig=0.15, seeds=4, cycles=8)
    out["kmc"].append(dict(i=i, phi=p["phi"], f=p["f"], sec=time.perf_counter() - t0)); print("KMC", i, round(out["kmc"][-1]["sec"], 2), flush=True)
tk = float(np.median([k["sec"] for k in out["kmc"]])); tkm = float(np.mean([k["sec"] for k in out["kmc"]]))
out["kmc_point_median_s"] = tk; out["kmc_point_mean_s"] = tkm; out["kmc_per_aggregate_s"] = tk / 4
out["dataset_1200_core_h"] = 1200 * tkm / 3600; out["exhaustive_grid_6561_core_h"] = 6561 * tkm / 3600
rng = np.random.default_rng(0); idx = rng.permutation(len(M.y)); n = int(0.8 * len(M.y)); tr = idx[:n]
LIMIT, f = 4.77e8, 5e4; d = np.linspace(10, 18, 81); K = np.logspace(np.log10(8e3), np.log10(3.4e4), 81); Dd, KK = np.meshgrid(d, K, indexing="ij")
Xg = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(LIMIT / f)), np.full(Dd.size, 0.2)])
MK = {"ANN": M.Ens, "GBT": M.gbt, **B.MODELS}
for nm, mk in MK.items():
    tf, tp = [], []
    for rep in range(3):
        t0 = time.perf_counter(); m = mk().fit(M.X[tr], M.y[tr]); t1 = time.perf_counter(); m.predict(Xg); t2 = time.perf_counter()
        tf.append(t1 - t0); tp.append(t2 - t1)
    out["models"][nm] = dict(t_fit_s=float(np.median(tf)), t_grid_s=float(np.median(tp)), t_pred_us=float(np.median(tp)) / len(Xg) * 1e6,
                             speedup_vs_kmc=tk / (float(np.median(tp)) / len(Xg)))
    print(nm, {k: round(v, 4) for k, v in out["models"][nm].items()}, flush=True)
out["_machine"] = dict(cpu=platform.processor() or "x86_64", cores_used=1, python=platform.python_version())
try: out["_machine"]["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except Exception: pass
json.dump(out, open("../data/time_all.json", "w"), indent=1)
print("KMC point", round(tk, 2), "s; dataset", round(out["dataset_1200_core_h"], 2), "core-h; exhaustive grid", round(out["exhaustive_grid_6561_core_h"], 1), "core-h")
