#!/usr/bin/env python3
"""
FedAvg with record-level differential privacy (DP-SGD in every client) and the effect on explanation drift.

Each client runs 18 full-batch steps per round for 20 rounds. In every step the per-example gradient of the
class-weighted logistic loss is clipped to L2 norm C, the clipped gradients are summed, Gaussian noise N(0,(zC)^2) is
added to each coordinate, and the sum is divided by the (public) sum of weights. Clients hold disjoint records, so the
privacy loss of a record is that of its own client: T x E = 360 compositions of a Gaussian mechanism with sensitivity C
and no subsampling. Renyi DP of one step is alpha/(2 z^2); over 360 steps and converted to (epsilon, delta) with
delta = 1e-5:   epsilon = min_alpha 360 alpha/(2 z^2) + log(1/delta)/(alpha - 1)   (alpha = 1 + sqrt(2 z^2 log(1/delta)/360)).
The noise multiplier z is solved for each target epsilon. The class counts used for the class weights are treated as public.
Logistic regression only. Reference: pooled non-private logistic regression, exact linear SHAP, shared baseline,
declined applicants with score >= 0.30, noise floor from 50 bootstrap retrainings.

Usage: python dp_fedavg.py    -> results_drift/dp_*.csv
"""
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
import explanation_drift as ed

OUT = ed.OUT; XTR, YTR = ed.Xtr, ed.ytr
CLIP, DELTA, STEPS, ROUNDS, EPOCHS, LR_ = 2.0, 1e-5, 360, 20, 18, 0.5
EPS_GRID = [1, 2, 4, 8, 16, 32]
model = v2.MODELS["LR"]


def eps_of(z):
    if z <= 0: return np.inf
    a = 1 + np.sqrt(2 * z * z * np.log(1 / DELTA) / STEPS)
    return STEPS * a / (2 * z * z) + np.log(1 / DELTA) / (a - 1)


def z_for(eps):
    lo, hi = 0.05, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if eps_of(mid) > eps: lo = mid
        else: hi = mid
    return hi


def dp_local(p, X, y, sw, z, rng):
    s = sw.sum(); l2 = 1.0
    for _ in range(EPOCHS):
        pr = model.prob(p, X)
        Xa = np.c_[X, np.ones(len(X))]
        G = ((pr - y) * sw)[:, None] * Xa
        nrm = np.linalg.norm(G, axis=1, keepdims=True)
        G = G * np.minimum(1, CLIP / np.maximum(nrm, 1e-12))
        g = G.sum(0) + rng.normal(0, z * CLIP, G.shape[1])
        g = g / s
        g[:-1] += l2 * p[:-1] / len(y)
        p = p - LR_ * g
    return p


def dp_fedavg(clients, z, rng):
    p = model.init()
    sws = [v2.classweights(c[1]) for c in clients]
    n = np.array([len(c[1]) for c in clients], float); w = n / n.sum()
    for _ in range(ROUNDS):
        p = np.sum([wi * dp_local(p, Xk, yk, sw, z, rng) for wi, (Xk, yk, _), sw in zip(w, clients, sws)], 0)
    return p


def jac_top5(A, B):
    ta, tb = ed.topm(A), ed.topm(B)
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)]))


def main():
    pref = ed.train_central(model, XTR, YTR)
    declined = XTR[model.prob(pref, XTR) >= ed.T_DEFAULT]
    Aref = model.attribution(pref, declined, ed.SHARED)
    fl = []
    for b in range(ed.B):
        r = np.random.default_rng(10_000 + b); idx = r.integers(0, len(YTR), len(YTR))
        fl.append(jac_top5(model.attribution(ed.train_central(model, XTR[idx], YTR[idx]), declined, ed.SHARED), Aref))
    floor = dict(mean=np.mean(fl), p05=np.quantile(fl, 0.05))
    print("floor", floor, "declined", len(declined))
    rows = []
    for regime in ["medium", "high"]:
        for seed in ed.SEEDS:
            cl = ed.partition(regime, seed)
            settings = [("non-private", 0.0, np.inf), ("clip only", 0.0, np.inf)] + [(f"eps={e}", z_for(e), e) for e in EPS_GRID]
            for label, z, e in settings:
                rng = np.random.default_rng(100 + seed)
                if label == "non-private":
                    p = v2.federated(model, cl, "fedavg")
                else:
                    p = dp_fedavg(cl, z, rng)
                rows.append(dict(regime=regime, seed=seed, setting=label, epsilon=e, z=z, test_auc=roc_auc_score(v2.yte, model.prob(p, v2.Xte)),
                                 overlap=jac_top5(model.attribution(p, declined, ed.SHARED), Aref),
                                 top1=float((ed.topm(model.attribution(p, declined, ed.SHARED))[:, 0] == ed.topm(Aref)[:, 0]).mean())))
        print("done", regime, flush=True)
    df = pd.DataFrame(rows); df["floor_p05"] = floor["p05"]; df["floor_mean"] = floor["mean"]
    df.to_csv(f"{OUT}/dp_fedavg.csv", index=False)
    g = df.groupby(["regime", "setting"]).agg(epsilon=("epsilon", "first"), z=("z", "first"), auc=("test_auc", "mean"), overlap=("overlap", "mean"),
                                              sd=("overlap", "std"), below=("overlap", lambda s: float((s < floor["p05"]).mean()))).reset_index()
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
