#!/usr/bin/env python3
"""Held-out group gaps of the pooled-data reference model and FedAvg (high heterogeneity, partition seed 0),
computed with the same models, partition and training budget as the explanation-drift study.
Run from this folder: python fairness_check.py  -> results_drift/e7_fairness.csv"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import explanation_drift as ed
import federated_xai_experiment_v2 as v2

Xte, yte = v2.Xte, v2.yte
mte = v2.meta.iloc[v2.te].reset_index(drop=True)
THR, MIN_N = 0.30, 5
rows = []
for mn, model in ed.MODELS.items():
    pref = ed.train_central(model, ed.Xtr, ed.ytr)
    cl = ed.partition("high", 0)
    fed = v2.federated(model, cl, "fedavg")
    for name, p in [("pooled", pref), ("fedavg", fed)]:
        pr = model.prob(p, Xte); pred = (pr >= THR).astype(int)
        for attr in ["sector", "division", "location_type", "owner_education"]:
            g = mte[attr].values; dp, fnr, fpr, sizes = [], [], [], []
            for lev in pd.unique(g):
                m = g == lev
                if m.sum() < MIN_N: continue
                yy, pp = yte[m], pred[m]; pos, neg = yy == 1, yy == 0
                dp.append(pp.mean()); sizes.append(int(m.sum()))
                fnr.append(((pp == 0) & pos).sum() / max(pos.sum(), 1)); fpr.append(((pp == 1) & neg).sum() / max(neg.sum(), 1))
            rows.append(dict(model=mn, setting=name, attribute=attr, groups=len(sizes), min_group=min(sizes), max_group=max(sizes),
                             DP_gap=max(dp) - min(dp), EO_gap=max(fnr) - min(fnr), PE_gap=max(fpr) - min(fpr)))
out = pd.DataFrame(rows); out.to_csv(f"{ed.OUT}/e7_fairness.csv", index=False)
print(out.round(3).to_string())
