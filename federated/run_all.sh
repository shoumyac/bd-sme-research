#!/usr/bin/env bash
# Reproduce every result in federated/. Takes about 10 minutes on a laptop CPU.
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
python summarise_drift.py                          # summary tables                                  -> results_drift/summary_*.csv
