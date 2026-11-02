#!/usr/bin/env python3
"""
Rank-graduation accuracy, robustness and explainability (RGA, RGR, RGE) and the integrated
compliance score of Giudici and Kolesnikov (2026, Machine Learning with Applications 23, 100821),
applied to pooled, federated and isolated models.

Definitions (their Eqs. 4, 5-7, 8-9, 10-12, 13-17), for n cases, predictions yhat and actuals y:
  RGA   = [sum_i i*y_(order by yhat, ascending) - sum_i i*y_(descending)] / [sum_i i*y_(ascending) - sum_i i*y_(descending)]
          (equals ROC-AUC for a binary outcome)
  RGR   = same construction with the original predictions yhat ordered by the ranks of the predictions
          computed on perturbed inputs
  RGE*  = same construction with the full model's predictions ordered by the ranks of the predictions
          of a model without the removed variables (1 with nothing removed, 0.5 with all removed)
Curves: RGR over the normalised Gaussian input perturbation (0 to 0.5 standard deviations) and RGE* over the
fraction of variables removed. Each curve is sampled at N_PTS points; AURGR and AURGE are the mean heights.
Accuracy enters as the point value RGA (= ROC-AUC on the held-out rows), not as the AURGA curve, because the
partial-RGA decomposition behind AURGA is not fully specified in the paper and we do not reproduce it.
The integrated compliance score is the mean over the grid of M(j,k) = mean(RGA, RGR_j, RGE_k) (arithmetic) or
cbrt(RGA * RGR_j * RGE_k) (geometric). It is therefore NOT numerically comparable with the scores in that paper.

Variable removal needs no retraining here: a removed variable is replaced by its training mean (0 after
standardisation for numeric columns), and variables are removed from least to most important by the
model's own mean absolute attribution. This is an occlusion version of the paper's reduced-model definition.

Run after explanation_drift.py:  python safe_metrics.py   ->  results_drift/e8_safe.csv
"""
import time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
import explanation_drift as ed

OUT = ed.OUT
N_PTS = 20            # points per curve
ALPHAS = np.linspace(0.0, 0.5, N_PTS)     # Gaussian perturbation, in standard deviations (features are standardised)
REPS = 5
t0 = time.time()


def _order_stats(y_sorted_by):
    n = len(y_sorted_by); i = np.arange(1, n + 1)
    return i, (i * y_sorted_by)


def rg(y, order_by):
    """Rank-graduation ratio of the values y when they are ordered by `order_by` (ascending), against the
    best (ascending y) and worst (descending y) orderings. Returns the ratio and the per-position terms."""
    n = len(y); i = np.arange(1, n + 1)
    ys = np.sort(y)
    asc = (i * ys).sum(); desc = (i * ys[::-1]).sum()
    ordered = y[np.argsort(order_by, kind="stable")]
    num_terms = i * ordered - i * ys[::-1]
    den = asc - desc
    return num_terms.sum() / den, num_terms / den


def perturbed_rgr(model, p, X, rng, alpha):
    yhat = model.prob(p, X)
    vals = []
    for _ in range(REPS if alpha > 0 else 1):
        yp = model.prob(p, X + rng.normal(0, alpha, X.shape)) if alpha > 0 else yhat
        r, _ = rg(yhat, yp)
        vals.append(r)
    return float(np.mean(vals))


def rge_curve(model, p, X, ref_X, base, n_pts=N_PTS):
    """RGE* as variables are removed (least important first) by replacing them with their training mean."""
    imp = np.abs(model.attribution(p, ref_X, base)).mean(0)
    order = np.argsort(imp)                       # least important first
    yhat = model.prob(p, X)
    steps = np.unique(np.linspace(0, X.shape[1], n_pts).astype(int))
    vals = []
    for s in steps:
        Xr = X.copy(); Xr[:, order[:s]] = base[order[:s]]
        r, _ = rg(yhat, model.prob(p, Xr))
        vals.append(r)
    vals = np.interp(np.linspace(0, X.shape[1], n_pts), steps, vals)
    return vals


def score_grid(rga, r, e):
    R = r[:, None]; E = e[None, :]
    arith = ((rga + R + E) / 3).mean()
    geo = np.cbrt(max(rga, 0) * np.clip(R, 0, None) * np.clip(E, 0, None)).mean()
    return float(arith), float(geo)


def evaluate(model, p, Xte, yte, ref_X, base, locs=None):
    ps = [p] if locs is None else locs
    out = []
    for q in ps:
        yhat = model.prob(q, Xte)
        rga0 = rg(yte.astype(float), yhat)[0]
        rng = np.random.default_rng(7)
        r_curve = np.array([perturbed_rgr(model, q, Xte, rng, a) for a in ALPHAS])
        e_curve = rge_curve(model, q, Xte, ref_X, base)
        ar, ge = score_grid(rga0, r_curve, e_curve)
        out.append(dict(rga=rga0, aurgr=r_curve.mean(), aurge=e_curve.mean(), score_arith=ar, score_geo=ge,
                        auc_check=roc_auc_score(yte, yhat)))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main():
    Xtr, ytr, Xte, yte = v2.Xtr, v2.ytr, v2.Xte, v2.yte
    base = Xtr.mean(0)
    rows = []
    for mn, model in ed.MODELS.items():
        pref = ed.train_central(model, Xtr, ytr)
        r = evaluate(model, pref, Xte, yte, Xtr, base)
        assert abs(r["rga"] - r["auc_check"]) < 1e-9, "RGA must equal ROC-AUC for a binary outcome"
        rows.append(dict(model=mn, source="centralised", seed=-1, **r))
        for seed in ed.SEEDS:
            cl = ed.partition("high", seed)
            locs = [v2.local_train(model, model.init(), Xk, yk, 300, 0.5) for Xk, yk, _ in cl]
            for a in ed.AGGS:
                p = v2.federated(model, cl, a)
                rows.append(dict(model=mn, source=a, seed=seed, **evaluate(model, p, Xte, yte, Xtr, base)))
            rows.append(dict(model=mn, source="isolated", seed=seed, **evaluate(model, None, Xte, yte, Xtr, base, locs=locs)))
        print(f"[{time.time()-t0:.0f}s] safe metrics {mn}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/e8_safe_by_seed.csv", index=False)
    g = df.groupby(["model", "source"]).agg(**{c: (c, "mean") for c in ["rga", "aurgr", "aurge", "score_arith", "score_geo"]},
                                           score_arith_sd=("score_arith", "std")).reset_index()
    g.to_csv(f"{OUT}/e8_safe.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
