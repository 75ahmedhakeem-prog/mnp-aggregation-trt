import json, numpy as np, prostate as P
SAR_PLAN = 12.5 * (4.5 / 10) ** 2          # W/gFe, clinical planning value scaled to the tolerated 4.5 kA/m (Johannsen 2005)
MEAS = dict(T20=41.1, T50=40.8, T90=40.1)  # Johannsen 2007 medians
W = [1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3]; SEEDS = range(8); out = []
for w in W:
    for sd in SEEDS:
        r = P.run(SAR_PLAN, seed=sd, w=w); t = r["T"][r["pro"]] - 37.0
        row = dict(w=w, seed=sd, dT20=float(np.percentile(t, 80)), dT50=float(np.percentile(t, 50)), dT90=float(np.percentile(t, 10)), dTmax=float(t.max()))
        row.update({f"F_{k}": (MEAS[k] - 37.0) / row["d" + k] for k in MEAS}); out.append(row)
json.dump(dict(SAR_plan=SAR_PLAN, meas=MEAS, rows=out), open("../data/prostate_sweep.json", "w"), indent=1)
for w in W:
    rs = [r for r in out if r["w"] == w]
    print(w, "fluid T20/T50/T90:", [round(37 + np.mean([r["d" + k] for r in rs]), 2) for k in ("T20", "T50", "T90")],
          "F needed:", [round(np.mean([r["F_" + k] for r in rs]), 2) for k in ("T20", "T50", "T90")])
