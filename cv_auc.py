# Federated Explainability Under Distribution Shift — code

Reproduction code for the paper **"Federated Explainability Under Distribution Shift:
Benchmarking SHAP Stability in Credit Risk Models Across Heterogeneous Emerging Market
Clients"** (Chowdhury, Das, Paul), submitted to *Machine Learning* (Springer),
Special Issue on Advances in Federated Learning for Critical Applications.

This extends the dataset and centralised analysis archived in this same repository.

## Data
Uses `bangladesh_sme_1200_records.csv` (1,200 anonymised SME loan applications,
CC-BY 4.0), included in this repository / Zenodo record. Place it where the scripts
expect it (`./data/` or alongside the script; the loader checks both).

## Scripts (run with Python 3.10+, `pip install numpy pandas scipy scikit-learn shap matplotlib`)
- `federated_xai_experiment.py` — horizontal FL (FedAvg/FedProx), cross-client SHAP
  stability, Dirichlet stress test, fairness; writes `results_federated/`.
- `federated_xai_experiment_v2.py {zoo|grid|extra}` — 10-model comparison; two federated
  families (logistic regression + MLP) across five aggregators; client-count sweep; DP
  ablation; writes `results_federated_v2/`.
- `federated_xai_experiment_v3.py {vfl|pfl}` — vertical and personalised federation.
- `cv_auc.py`, `cv_extra.py` — five-fold cross-validated ROC-AUC for all settings.
- `mitigation.py` — shared-baseline explanation mitigation and the Proposition check.
- `federated_xai_experiment.ipynb` — Colab-ready notebook for the horizontal-FL pipeline.

All randomness is seeded (`random_state = 42`). Licence: MIT (see repository LICENSE).
