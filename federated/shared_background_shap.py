#!/usr/bin/env python3
"""
Head-to-head: how a federated model is explained (background data) versus how it is trained.

Model-agnostic KernelSHAP depends on a background dataset. In a federation each institution would naturally use its own
rows (local background), which makes explanations of the same global model differ between institutions. A consistent
alternative \\citep{ducange2024fuzz,ducange2026fedshap} is a shared background built without moving rows: here a
federated k-means (each client sends cluster sums and counts, never rows). Three ways of explaining the FedAvg model
are compared with the explanation of the pooled-data reference model against a pooled background of 20 training rows:
  fed_pooledbg    FedAvg model, same pooled background as the reference (isolates the model difference)
  fed_localbg     FedAvg model, each applicant explained against 20 rows of its own client (common practice)
  fed_sharedbg    FedAvg model, one shared background of 20 federated k-means centroids
Controls: bootstrap retrainings of the pooled model explained against the pooled background (noise floor), and the
reference model explained against a second random pooled background (pure background-sampling noise).
Top-5 Jaccard overlap on 150 applicants sampled from those the reference declines (score >= 0.30), 5 partition seeds.

Usage: python shared_background_shap.py     -> results_drift/sbg_*.csv
"""
import time, warnings
import numpy as np, pandas as pd, shap
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
import explanation_drift as ed

OUT = ed.OUT; XTR, YTR = ed.Xtr, ed.ytr
NBG, NEXPL, NSAMP, NBOOT, T = 20, 150, 1500, 20, 0.30
t0 = time.time()


def kshap(model, p, X, bg):
    f = lambda Z: model.prob(p, Z)
    return np.asarray(shap.KernelExplainer(f, bg).shap_values(X, nsamples=NSAMP, silent=True))


def fed_kmeans(clients, K=NBG, iters=8, seed=0):
    """Federated Lloyd iterations: clients return per-centroid sums and counts only."""
    rng = np.random.default_rng(seed)
    mu = np.vstack([c[0] for c in clients]).mean(0)
    sd = np.vstack([c[0] for c in clients]).std(0) + 1e-9
    C = mu + rng.normal(0, 1, (K, len(mu))) * sd              # server-side random initial centroids
    for _ in range(iters):
        S = np.zeros_like(C); N = np.zeros(K)
        for Xk, _, _ in clients:
            a = ((Xk[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
            for k in range(K):
                m = a == k
                S[k] += Xk[m].sum(0); N[k] += m.sum()
        C = np.where(N[:, None] > 0, S / np.maximum(N[:, None], 1), C)
    return C


def top5(A): return np.argsort(-np.abs(A), 1)[:, :5]


def overlap(A, B):
    ta, tb = top5(A), top5(B)
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)]))


def main():
    rows, floors = [], []
    rng0 = np.random.default_rng(0)
    bg_pool = XTR[rng0.choice(len(XTR), NBG, replace=False)]
    bg_pool2 = XTR[rng0.choice(len(XTR), NBG, replace=False)]
    for mn, model in ed.MODELS.items():
        pref = ed.train_central(model, XTR, YTR)
        boots = []
        for b in range(NBOOT):
            r = np.random.default_rng(10_000 + b); idx = r.integers(0, len(YTR), len(YTR))
            boots.append(ed.train_central(model, XTR[idx], YTR[idx]))
        dec_all = np.where(model.prob(pref, XTR) >= T)[0]
        sel = np.sort(np.random.default_rng(1).choice(dec_all, NEXPL, replace=False))
        Xs = XTR[sel]
        Aref = kshap(model, pref, Xs, bg_pool)
        fl = [overlap(kshap(model, pb, Xs, bg_pool), Aref) for pb in boots]
        ctrl = overlap(kshap(model, pref, Xs, bg_pool2), Aref)
        floors.append(dict(model=mn, floor_mean=np.mean(fl), floor_p05=np.quantile(fl, 0.05), bg_noise_only=ctrl, n_expl=NEXPL))
        print(f"[{time.time()-t0:.0f}s] {mn}: floor {np.mean(fl):.3f} ({np.quantile(fl,0.05):.3f}), background-only noise {ctrl:.3f}", flush=True)
        for regime in ["medium", "high"]:
            for seed in ed.SEEDS:
                cl = ed.partition(regime, seed)
                pfl = v2.federated(model, cl, "fedavg")
                # client of every selected applicant
                owner = np.full(len(XTR), -1)
                for k, (Xk, _, _) in enumerate(cl):
                    # rows of XTR belonging to client k (match by value; partition returns row copies)
                    pass
                key = {tuple(np.round(x, 9)): i for i, x in enumerate(XTR)}
                for k, (Xk, _, _) in enumerate(cl):
                    for x in Xk:
                        owner[key[tuple(np.round(x, 9))]] = k
                A_pool = kshap(model, pfl, Xs, bg_pool)
                A_loc = np.zeros_like(A_pool)
                for k, (Xk, _, _) in enumerate(cl):
                    m = owner[sel] == k
                    if m.any():
                        bgk = Xk[np.random.default_rng(seed + k).choice(len(Xk), min(NBG, len(Xk)), replace=False)]
                        A_loc[m] = kshap(model, pfl, Xs[m], bgk)
                A_sh = kshap(model, pfl, Xs, fed_kmeans(cl, seed=seed))
                for nm, A in [("fed_pooledbg", A_pool), ("fed_localbg", A_loc), ("fed_sharedbg", A_sh)]:
                    rows.append(dict(model=mn, regime=regime, seed=seed, setting=nm, overlap=overlap(A, Aref), n_expl=NEXPL))
                # consistency across institutions: same applicants explained by each client's own background
                print(f"[{time.time()-t0:.0f}s] {mn} {regime} seed {seed}: pooled {rows[-3]['overlap']:.3f} local {rows[-2]['overlap']:.3f} shared {rows[-1]['overlap']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/sbg_fl_vs_reference.csv", index=False)
        pd.DataFrame(floors).to_csv(f"{OUT}/sbg_noise_floor.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
