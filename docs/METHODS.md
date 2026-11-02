# Methods: explanation drift under federation

This note defines the analysis in `federated/explanation_drift.py`. All code is seeded and every number in `federated/results_drift/` is produced by it.

## Setting

* **Data.** The 1,200 loans; a stratified 80/20 split with `random_state = 42` gives 960 training rows (the federated pool) and 240 test rows (held-out ROC-AUC only).
* **Inputs.** 44 columns after one-hot encoding (see `data/DATA_DICTIONARY.md`).
* **Models.** A logistic regression (explained with exact linear SHAP values, `phi_j = w_j (x_j - baseline_j)`) and a one-hidden-layer network with 32 ReLU units (explained with Integrated Gradients, 12 steps). Both are trained with class-balanced weights, so their scores are not calibrated probabilities.
* **Clients.** Three archetypes built from location type, sector and division: rural and agricultural borrowers, urban borrowers in manufacturing and services, and borrowers concentrated in three divisions. Heterogeneity is `low`, `medium` or `high`; the `high` partition is deterministic apart from rows that match no archetype. A second family of partitions uses Dirichlet label skew (alpha 0.1, 0.5, 5.0). All records come from one bank and the central bank, so the archetypes are stylised and do not correspond to real institutions.
* **Federation.** FedAvg, FedProx (mu = 0.1), SCAFFOLD and coordinate-wise median, 20 rounds of 18 local full-batch gradient steps. "Isolated" models are trained on one client's data only and never exchange parameters.

## Reference model and drift

* **Reference.** The same model class trained on all 960 rows (400 full-batch gradient steps).
* **Instances.** Only rows the reference model declines, that is a reference score of at least 0.30 (sensitivity: 0.20, 0.40, 0.50). Only declined applicants receive an adverse-action notice.
* **Baseline.** Both models are explained against the same baseline, so differences come from the models. Primary analysis: the mean of the pooled training data (`shared`). Sensitivity: each client's own mean (`client`).
* **Agreement metrics, per declined applicant, averaged over applicants.**
  * `jac`: Jaccard overlap of the five largest-magnitude attributions (the five reasons cited).
  * `top1`: share of applicants whose single largest attribution is the same feature.
  * `cos`: cosine similarity of the two attribution vectors.
  * `exact`: share with identical top-5 sets. This is the strict criterion used in earlier drafts; `legacy_violation = 1 - exact` is reported for comparison only.

## Retraining noise floor

Fifty bootstrap resamples of the 960 training rows are drawn, the model is retrained on each, and its explanations are compared with the reference in exactly the same way (`boot`). For the network a second floor also changes the random initialisation (`boot_init`). The floor shows how far explanations move when a lender simply retrains on a refreshed sample, which is an everyday event nobody treats as a defect. It is a tolerance benchmark, not a null distribution: federated models are trained on the same rows as the reference, whereas bootstrap models see a resample. A federated result is called *within the floor* when its agreement is at least the 5th percentile of the bootstrap agreements.

## Other analyses

* `e2_rank_agreement_sources.csv`: cross-client agreement of global feature rankings (mean pairwise Spearman correlation of mean absolute attributions), for the pooled model and FedAvg, under per-client and shared baselines. Separates data heterogeneity and baseline choice from aggregation.
* `e3_stress.csv`: Dirichlet label skew, number of clients (3, 5, 10), Gaussian noise added to the aggregate (logistic regression only) and the decline threshold.
* `federated_xai_experiment_v3.py`: vertical federation (44 features held by three parties) and personalised federation (global model fine-tuned on each client). These change the structure of the explanation problem and are reported separately.
* `mitigation.py`: the shared-baseline explanation and a check of the dispersion proposition for linear models, `std_k(phi_j) = |w_j| std_k(mu_j)`.

## Limitations

One dataset from one bank and the central bank; simulated clients; 960 training rows; the network is explained with Integrated Gradients and the linear model with SHAP, so the two model families are compared on trends only; no human study of whether explanation differences change decisions.
