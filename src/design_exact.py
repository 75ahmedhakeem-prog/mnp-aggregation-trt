"""Exact (KMC) versus surrogate design of the dense-aggregate particle (objective: KMC SAR at phi = 0.2 at the H*f limit),
for the two ANN searches of the paper: 50 kHz (grid of ml_design.py) and the stimulation-safe 85 kHz (grid of ml_design_safe.py).
 (a) exact refinement of the surrogate: the 10 best ANN candidates, thinned by non-maximum suppression (no two candidates
     within 3 grid steps in both d and log K), are evaluated with KMC (16 aggregates, 10 cycles, as ml_verify2.py); for top-K
     (K = 1, 3, 5, 10) the best KMC SAR and the regret are reported;
 (b) exact coarse search without surrogate: KMC (4 aggregates, 8 cycles, as gen_ml.py) on a 9 x 9 (d, K) grid spanning the same
     box; its best point is re-evaluated with 16 aggregates.
The exact reference is the best 16-aggregate KMC SAR found by (a) or (b); regret = reference - KMC SAR of the method's choice.
The chosen designs are also evaluated in the dispersed and soft (phi = 0.1) states. Run inside src/ (2 worker processes)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import json, time, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG = 4.77e8, 0.724, 0.15
CASES = {"50kHz": dict(f=5e4, d=np.linspace(10, 18, 81), K=np.logspace(np.log10(8e3), np.log10(3.4e4), 81)),
         "85kHz": dict(f=8.5e4, d=np.linspace(9.5, 18, 86), K=np.logspace(np.log10(8e3), np.log10(4e4), 81))}

def kmc(a):
    case, d, K, ph, seeds, cycles = a; f = CASES[case]["f"]; t0 = time.perf_counter()
    m, e = S.sar("dilute" if ph is None else "loose", ph, d, K, LIMIT / f, f, sig=SIG, seeds=seeds, cycles=cycles)
    return dict(case=case, d=float(d), K=float(K), phi=ph, seeds=seeds, SAR_Fe=m / FE, se_Fe=e / FE, sec=time.perf_counter() - t0)

def ann_candidates(m, M, case, n=10, sep=3):
    c = CASES[case]; f = c["f"]; H = LIMIT / f; Dd, KK = np.meshgrid(c["d"], c["K"], indexing="ij")
    X = np.column_stack([Dd.ravel(), np.log10(KK.ravel()), np.full(Dd.size, 3.75), np.full(Dd.size, np.log10(f)), np.full(Dd.size, np.log10(H)), np.full(Dd.size, 0.2)])
    s = (10 ** m.predict(X) - M.SHIFT).reshape(Dd.shape); picked = []
    for k in np.argsort(s.ravel())[::-1]:
        i, j = np.unravel_index(k, s.shape)
        if all(abs(i - a) > sep or abs(j - b) > sep for a, b in picked): picked.append((i, j))
        if len(picked) == n: break
    return [dict(rank=r + 1, d=float(c["d"][i]), K=float(c["K"][j]), ANN_02=float(s[i, j])) for r, (i, j) in enumerate(picked)], int(s.size)

def find(R, case, d, K, ph, seeds):
    return next(x for x in R if x["case"] == case and x["d"] == d and x["K"] == K and x["phi"] == ph and x["seeds"] == seeds)

if __name__ == "__main__":
    import ml_sar as M
    T0 = time.perf_counter(); m = M.Ens().fit(M.X, M.y); cand = {}; ngrid = {}
    for case in CASES: cand[case], ngrid[case] = ann_candidates(m, M, case)
    t_ann = time.perf_counter() - T0; print("ANN fit + candidates", round(t_ann, 1), "s", flush=True)
    coarse = {case: list(itertools.product(np.linspace(c["d"][0], c["d"][-1], 9), np.logspace(np.log10(c["K"][0]), np.log10(c["K"][-1]), 9))) for case, c in CASES.items()}
    jobs = [(case, c["d"], c["K"], 0.2, 16, 10) for case in CASES for c in cand[case]] + [(case, d, K, 0.2, 4, 8) for case in CASES for d, K in coarse[case]]
    t1 = time.perf_counter()
    with Pool(2) as p: R = p.map(kmc, jobs, chunksize=1)
    # re-evaluate the coarse-grid winners with 16 aggregates
    cbest = {case: max(coarse[case], key=lambda q: find(R, case, q[0], q[1], 0.2, 4)["SAR_Fe"]) for case in CASES}
    with Pool(2) as p: R += p.map(kmc, [(case, *cbest[case], 0.2, 16, 10) for case in CASES])
    res = {}
    for case in CASES:
        for c in cand[case]:
            r = find(R, case, c["d"], c["K"], 0.2, 16); c.update(KMC_02=r["SAR_Fe"], se_02=r["se_Fe"])
        g = find(R, case, *cbest[case], 0.2, 16); g4 = find(R, case, *cbest[case], 0.2, 4)
        best = max([c["KMC_02"] for c in cand[case]] + [g["SAR_Fe"]]); topk = {}
        for k in (1, 3, 5, 10):
            c = max(cand[case][:k], key=lambda c: c["KMC_02"])
            topk[str(k)] = dict(rank=c["rank"], d=c["d"], K=c["K"], ANN_02=c["ANN_02"], KMC_02=c["KMC_02"], se_02=c["se_02"], regret=best - c["KMC_02"],
                                rel_regret=(best - c["KMC_02"]) / best, kmc_calls=k)
        res[case] = dict(f=CASES[case]["f"], H=LIMIT / CASES[case]["f"], n_grid=ngrid[case], candidates=cand[case], topk=topk,
                         coarse=[dict(d=float(d), K=float(K), SAR_02=find(R, case, d, K, 0.2, 4)["SAR_Fe"]) for d, K in coarse[case]],
                         coarse_best=dict(d=g["d"], K=g["K"], SAR_02_4agg=g4["SAR_Fe"], KMC_02=g["SAR_Fe"], se_02=g["se_Fe"], regret=best - g["SAR_Fe"], kmc_calls=82),
                         best_exact=best, ann_top1_overestimate=cand[case][0]["ANN_02"] / cand[case][0]["KMC_02"] - 1)
    t_kmc = time.perf_counter() - t1
    # the refined (top-10) and coarse winners in the dispersed and soft states
    fin = [(case, res[case]["topk"]["10"]["d"], res[case]["topk"]["10"]["K"], ph, 16, 10) for case in CASES for ph in (None, 0.1)]
    with Pool(2) as p: F = p.map(kmc, fin)
    for case in CASES:
        t = res[case]["topk"]["10"]
        t.update(KMC_disp=find(F, case, t["d"], t["K"], None, 16)["SAR_Fe"], KMC_01=find(F, case, t["d"], t["K"], 0.1, 16)["SAR_Fe"])
    s4 = [x["sec"] for x in R if x["seeds"] == 4]; s16 = [x["sec"] for x in R if x["seeds"] == 16]
    res["cost"] = dict(ann_fit_and_search_s=t_ann, kmc_wall_s_2proc=t_kmc, kmc_call_4agg_median_s=float(np.median(s4)), kmc_call_16agg_median_s=float(np.median(s16)),
                       note="per-call times measured with 2 worker processes on 2 cores; single-core reference timings are in time_all.json")
    json.dump(res, open("../data/design_exact.json", "w"), indent=1)
    for case in CASES:
        print(case, "coarse", {k: round(v, 3) for k, v in res[case]["coarse_best"].items()}, "overest", round(res[case]["ann_top1_overestimate"], 3))
        for k, v in res[case]["topk"].items(): print("  top", k, {q: round(x, 3) for q, x in v.items()})
