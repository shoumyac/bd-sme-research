#!/usr/bin/env python3
"""
Numerical check of Proposition 1 (baseline-driven attribution dispersion) for the FedAvg logistic regression.

Predicted dispersion of feature j:  |w_j| * std_k(mu_j^k)   (w = FedAvg weights, mu^k = client k baseline).
Measured (A): std over clients of the client-mean |attribution| under each client's own baseline (the quantity
              reported as 'attribution instability' in earlier tables).
Measured (B): std over clients of the mean signed attribution of one fixed instance set (the pooled training
              rows) explained under each client's baseline. By linearity this equals the prediction exactly.
High-heterogeneity partition, partition seeds 0 to 4 (the partition is deterministic apart from a few unassigned rows).
Writes results_drift/e2_proposition.csv and prints the correlations.
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import federated_xai_experiment_v2 as v2

Xtr, ytr, meta, feat = v2.Xtr, v2.ytr, v2.meta_tr, v2.feat
LR = v2.LRModel()
rows, cors = [], []
for seed in range(5):
    v2.RNG = np.random.default_rng(seed)
    cl = v2.partition(meta, Xtr, ytr, "high")
    pg = v2.federated(LR, cl, "fedavg"); w = pg[:-1]
    mus = np.array([Xk.mean(0) for Xk, _, _ in cl])
    pred = np.abs(w) * mus.std(0)
    A = np.array([np.abs(LR.attribution(pg, Xk, Xk.mean(0))).mean(0) for Xk, _, _ in cl]).std(0)
    B = np.array([LR.attribution(pg, Xtr, Xk.mean(0)).mean(0) for Xk, _, _ in cl]).std(0)
    cors.append(dict(seed=seed, pearson_A=pearsonr(pred, A)[0], spearman_A=spearmanr(pred, A)[0],
                     pearson_B=pearsonr(pred, B)[0], max_abs_diff_B=float(np.abs(pred - B).max())))
    if seed == 0:
        rows = pd.DataFrame(dict(feature=feat, abs_weight=np.abs(w), baseline_std=mus.std(0), predicted=pred,
                                 measured_A=A, measured_B=B))
rows.to_csv("results_drift/e2_proposition.csv", index=False)
c = pd.DataFrame(cors); print(c.round(4).to_string(index=False))
print("top predicted:", list(rows.sort_values("predicted", ascending=False).feature.head(5)))
