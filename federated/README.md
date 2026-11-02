# Federated experiments

Run everything with `bash run_all.sh` from this folder (under 10 minutes). Every result reproduces exactly except wall-clock timings (the `seconds` columns of `e4_*.csv` and `e6_joint.csv`), which vary from run to run. The scripts read `../data/bangladesh_sme_1200_records.csv` and write into the `results_*` folders next to them.

| Script | What it does | Output |
|---|---|---|
| `federated_xai_experiment.py` | Horizontal federation of a logistic regression (FedAvg, FedProx, isolated), exact SHAP, Dirichlet stress test, subgroup fairness | `results_federated/` |
| `federated_xai_experiment_v2.py zoo` | Ten centralised classifiers, 5-fold cross-validation | `results_federated_v2/model_zoo.csv` |
| `federated_xai_experiment_v2.py grid` | Logistic regression and MLP, three heterogeneity regimes, five aggregators | `results_federated_v2/fed_stability_v2.csv` |
| `federated_xai_experiment_v2.py extra` | Client scaling, differential-privacy noise, Dirichlet skew | `results_federated_v2/` |
| `federated_xai_experiment_v3.py vfl` / `pfl` | Vertical and personalised federation | `results_federated_v3/` |
| `mitigation.py` | Shared-baseline explanation and the dispersion proposition | `results_federated_v2/mitigation.csv`, `proposition_check.csv` |
| `cv_auc.py`, `cv_extra.py` | 5-fold cross-validated ROC-AUC for all settings | `results_federated_v2/cv_auc_*.csv`, `cv_extra_log.txt` |
| `explanation_drift.py` | Agreement with the pooled-data reference model for declined applicants, retraining noise floor, stress tests | `results_drift/e1_*.csv`, `e2_*.csv`, `e3_*.csv` |
| `proposition_check.py` | Numerical check of the dispersion proposition (predicted against measured cross-client spread) | `results_drift/e2_proposition.csv` |
| `summarise_drift.py` | Summary tables | `results_drift/summary_*.csv` |
| `cost_concentration.py` | Training and communication cost, explainer time, concentration of attributions, KernelSHAP and LIME agreement with the native explainer, joint accuracy-drift-robustness-cost view | `results_drift/e4_*.csv`, `e5_*.csv`, `e6_joint.csv` |
| `fairness_check.py` | Held-out group gaps (demographic parity, equal opportunity, predictive equality), pooled model against FedAvg | `results_drift/e7_fairness.csv` |
| `make_figures.py`, `make_tables.py` | Figures and LaTeX tables for the manuscript, drawn from the result files | `figures/`, `tables/` |

`federated_xai_experiment.ipynb` is a Colab-ready notebook of the first script. Definitions are in [`../docs/METHODS.md`](../docs/METHODS.md).

Columns in `summary_e1.csv`: `jac` (top-5 overlap with the reference), `top1` (same top reason), `cos` (attribution cosine), `exact` (identical top-5 sets), `legacy_violation` (1 - `exact`), `floor_*` (retraining noise floor), `within_floor_jac` (agreement at least the floor's 5th percentile), `n_seeds` (partition seeds; the high-heterogeneity partition is deterministic, so its standard deviation is zero).
