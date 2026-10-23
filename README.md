# mnp-aggregation-trt

Code and data for the manuscript **"Magnetic nanoparticle aggregation in tumour tissue can cancel most of the planned thermoradiotherapy benefit: a particle-to-outcome model"** (A.M. Abdel Hakeem, S.A. Ahmed).

The framework couples
1. a two-state Néel kinetic Monte Carlo (KMC) model of dynamic hysteresis with exact Stoner–Wohlfarth barriers and dipolar coupling (`kmc.py`, `sar_cluster.py`),
2. the field safety limit H·f ≤ 4.77 × 10⁸ A m⁻¹ s⁻¹,
3. a steady Pennes model of a prostate treated with 24 magnetic-fluid depots (`prostate.py`),
4. the extended linear-quadratic thermoradiobiology model (`trbio.py`, `therapy.py`),
5. KMC design maps and an ANN / gradient-boosted-tree surrogate (`design_map.py`, `gen_ml.py`, `ml_sar.py`, `ml_design.py`).

## Requirements
Python ≥ 3.10, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pillow` (see `requirements.txt`).

## Reproducing the results (run inside `src/`)
| Step | Script | Output |
|---|---|---|
| KMC vs linear response | `verify_lrt.py` | `data/verify_lrt.json` |
| Calibration on soft aggregates (García-Acevedo et al. 2025) | `calibrate.py` | `data/calib_soft.json` |
| Dense-aggregate validation | `validate_ga.py`, `predict.py` | `data/validate_ga.json`, `data/predict_phi.json` |
| Intracellular benchmark (Di Corato et al. 2014) | `dicorato.py` | `data/dicorato.json` |
| Chain / compact benchmark (Ortega-Julia et al. 2023) | `bench_oj.py` | `data/bench_oj.json` |
| Clinical particle and field dependence | `nanotherm.py K`, `nanotherm.py phi 1.4e4`, `nt_field.py` | `data/nanotherm_*.json`, `data/nt_field.json` |
| Prostate temperatures | `prostate_sweep.py` | `data/prostate_sweep.json` |
| Design maps at the field limit | `design_map.py`, `clin50.py` | `data/design_map.json`, `data/clin50.json` |
| Thermoradiotherapy | `step3_therapy.py` | `data/step3_therapy.json` |
| Surrogate dataset (1200 KMC runs) | `gen_ml.py` | `data/ml_dataset.json` |
| Surrogate models | `ml_sar.py` | `data/ml_sar.json` |
| Surrogate design + KMC check | `ml_design.py`, `ml_verify.py`, `ml_verify2.py` | `data/ml_design.json`, `data/ml_verify*.json` |
| Uncertainty, sensitivity and TCP | `maps_cache.py`, `uq_therapy.py` | `data/uq_therapy.json` (maps cached in `data/maps_cache.npz`, regenerated on first run) |
| Field scan around the limit | `field_scan.py` | `data/field_scan.json` |
| Nerve-stimulation index | `stim.py` | printout |
| Stimulation-safe design (85 kHz) | `ml_design_safe.py`, `safe_verify.py` | `data/ml_design_safe.json`, `data/safe_verify.json` |
| Progressive aggregation, required field, 85 kHz therapy | `step4_extra.py` | `data/step4_extra.json` |
| Packing fraction from clinical temperatures | `clin_inverse.py` | `data/clin_inverse.json` |
| Prostate-cell thermal ratio (Heo et al. 2024) | `heo_calib.py` | `data/heo_calib.json` |
| Gain versus packing fraction, mixed states | `extra_kmc.py`, `gain_phi.py` | `data/extra_kmc.json`, `data/gain_phi.json` |
| KMC convergence | `extra_kmc.py`, `conv2.py` | `data/extra_kmc.json`, `data/conv2.json` |
| Transient heating | `transient.py` | `data/transient.json` |
| Figures | `fig_*.py` | `figures/*.png` |

`data/validation_inventory.csv` lists the experimental and clinical data used for validation, and `data/ga2025_fig6.csv` the digitised aggregate data.

## Licence
MIT
