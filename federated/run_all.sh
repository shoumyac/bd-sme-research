#!/usr/bin/env bash
# Reproduce every result in federated/. Takes about 30 minutes on one CPU core (external-dataset audit about 7, KernelSHAP comparison about 11, boosted trees about 3, DP-FedAvg about 1).
# Run from anywhere; the scripts expect to be run from this folder (they read ../data/).
set -euo pipefail
cd "$(dirname "$0")"
python federated_xai_experiment.py                 # horizontal FL, Dirichlet stress test, fairness  -> results_federated/
python federated_xai_experiment_v2.py zoo          # ten centralised classifiers                     -> results_federated_v2/
python federated_xai_experiment_v2.py grid         # LR and MLP, 3 regimes, 5 aggregators            -> results_federated_v2/
python federated_xai_experiment_v2.py extra        # client scaling, DP noise, Dirichlet             -> results_federated_v2/
python federated_xai_experiment_v3.py vfl          # vertical federation                             -> results_federated_v3/
python federated_xai_experiment_v3.py pfl          # personalised federation                         -> results_federated_v3/
python mitigation.py                               # shared-baseline explanation and Proposition 1   -> results_federated_v2/
for s in low medium high; do python cv_auc.py "$s"; done   # 5-fold cross-validated ROC-AUC
python cv_extra.py | tee results_federated_v2/cv_extra_log.txt
python explanation_drift.py all                    # agreement with the pooled-data reference model  -> results_drift/
python proposition_check.py                        # numerical check of Proposition 1               -> results_drift/e2_proposition.csv
python summarise_drift.py                          # summary tables                                  -> results_drift/summary_*.csv
python cost_concentration.py                       # cost, concentration, explainer agreement, joint view -> results_drift/e4_*.csv, e5_*.csv, e6_joint.csv
python safe_metrics.py                             # rank-graduation accuracy, robustness, explainability and compliance score -> results_drift/e8_safe.csv
python fairness_check.py                           # held-out group gaps, pooled model vs FedAvg    -> results_drift/e7_fairness.csv
python fetch_external_data.py                    # downloads three public credit datasets into ../data/external/ (needs internet)
python external_datasets.py                        # audit on German credit, Taiwan card default, Taiwan bankruptcy -> results_drift/ext_*.csv
python federated_gbdt.py                           # federated histogram boosted trees + TreeSHAP    -> results_drift/gbdt_*.csv
python shared_background_shap.py                   # KernelSHAP pooled / local / shared federated background -> results_drift/sbg_*.csv
python dp_fedavg.py                                # differentially private FedAvg, epsilon sweep    -> results_drift/dp_fedavg.csv
python make_figures.py                             # figures for the manuscript                     -> figures/
python make_tables.py                              # LaTeX tables for the manuscript                -> tables/
