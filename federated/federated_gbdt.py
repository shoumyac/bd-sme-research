#!/usr/bin/env python3
"""
Federated gradient-boosted trees and explanation drift (TreeSHAP).

Horizontal federated histogram boosting in the style of SecureBoost / federated XGBoost: for every tree node the
clients send per-feature, per-bin sums of gradients and Hessians, the server adds them (optionally with Gaussian
noise), chooses the best split and returns it. Four ways of using the same clients are compared with a pooled
reference trained by the same algorithm on all rows:
  hist      federated histogram aggregation (lossless up to floating point when no noise is added)
  hist_s*   the same with Gaussian noise (sd = sigma x the root Hessian) added to every aggregated histogram entry
  bagging   every client trains its own boosted ensemble; the global model averages them, weighted by client size
  cyclic    trees are trained one after another on one client each (round robin), on that client's rows only,
            using the margin of the global ensemble built so far
  isolated  every client keeps its own ensemble
Explanations are exact path-dependent TreeSHAP values on the log-odds scale (shap.TreeExplainer on a tree dictionary);
a node's cover is the number of training rows of the training client(s) that reach it. Drift = top-5 reason overlap
with the pooled reference on the applicants it declines (riskiest 30% of training rows), against a noise floor from
50 bootstrap retrainings of the reference. Datasets: the Bangladeshi loans and the three public datasets.

Usage: python federated_gbdt.py    -> results_drift/gbdt_*.csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import shap
import federated_xai_experiment_v2 as v2
import explanation_drift as ed
import external_datasets as xd

OUT = ed.OUT
NB, DEPTH, NTREE, LRATE, LAM, MINH = 16, 3, 50, 0.1, 1.0, 1.0
SIGMAS = [0.005, 0.02, 0.05]
SEEDS = ed.SEEDS; B = ed.B; M = 5
t0 = time.time()


# ------------------------------------------------------------------ binning
def thresholds(X):
    thr = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) < 2:
            thr.append(np.array([])); continue
        mids = (u[:-1] + u[1:]) / 2
        if len(mids) > NB - 1:
            qs = np.quantile(X[:, j], np.linspace(0, 1, NB + 1)[1:-1])
            idx = np.unique(np.clip(np.searchsorted(u, qs, side="right") - 1, 0, len(mids) - 1))
            mids = mids[idx]
        thr.append(mids)
    return thr


def binned(X, thr):
    return np.stack([np.searchsorted(thr[j], X[:, j]) for j in range(X.shape[1])], 1)


def sig(z): return 1 / (1 + np.exp(-z))


def bal_w(y):
    n, n1 = len(y), y.sum()
    return np.where(y == 1, n / (2 * max(n1, 1)), n / (2 * max(n - n1, 1)))


# ------------------------------------------------------------------ one tree from client histograms
def fit_tree(Bm, G, H, thr, noise_sd=0.0, rng=None):
    """Bm, G, H: lists over participating clients. Returns a tree (arrays) and the leaf of every row."""
    d = Bm[0].shape[1]
    F = [bm + (np.arange(d) * NB)[None, :] for bm in Bm]
    nid = [np.zeros(len(b), int) for b in Bm]
    cl, cr, feat, th, val, cnt = [-1], [-1], [-2], [0.0], [0.0], [0.0]
    frontier = [0]
    for depth in range(DEPTH + 1):
        nxt = []
        for node in frontier:
            Gh = np.zeros(d * NB); Hh = np.zeros(d * NB); c = 0
            for i in range(len(Bm)):
                r = np.where(nid[i] == node)[0]
                c += len(r)
                if len(r):
                    idx = F[i][r].ravel()
                    Gh += np.bincount(idx, np.repeat(G[i][r], d), d * NB)
                    Hh += np.bincount(idx, np.repeat(H[i][r], d), d * NB)
            if noise_sd > 0:
                Gh = Gh + rng.normal(0, noise_sd, Gh.shape); Hh = Hh + rng.normal(0, noise_sd, Hh.shape)
            Gm, Hm = Gh.reshape(d, NB), Hh.reshape(d, NB)
            cnt[node] = float(c)
            Gt, Ht = Gm[0].sum(), Hm[0].sum()
            val[node] = -Gt / (Ht + LAM) * LRATE
            if depth == DEPTH or c < 2:
                continue
            GL, HL = np.cumsum(Gm, 1), np.cumsum(Hm, 1)
            Gs, Hs = Gm.sum(1, keepdims=True), Hm.sum(1, keepdims=True)
            GR, HR = Gs - GL, Hs - HL
            gain = GL ** 2 / (HL + LAM) + GR ** 2 / (HR + LAM) - Gs ** 2 / (Hs + LAM)
            ok = (HL >= MINH) & (HR >= MINH)
            for j in range(d):                      # only real thresholds
                ok[j, len(thr[j]):] = False
            gain = np.where(ok, gain, -np.inf)
            j, b = np.unravel_index(np.argmax(gain), gain.shape)
            if not np.isfinite(gain[j, b]) or gain[j, b] <= 1e-9:
                continue
            L, R = len(cl), len(cl) + 1
            for _ in range(2):
                cl.append(-1); cr.append(-1); feat.append(-2); th.append(0.0); val.append(0.0); cnt.append(0.0)
            cl[node], cr[node], feat[node], th[node] = L, R, int(j), float(thr[j][b])
            for i in range(len(Bm)):
                m = nid[i] == node
                left = Bm[i][m, j] <= b
                tmp = nid[i][m]; tmp[left] = L; tmp[~left] = R; nid[i][m] = tmp
            nxt += [L, R]
        frontier = nxt
        if not frontier:
            break
    tree = dict(children_left=np.array(cl), children_right=np.array(cr), children_default=np.array(cl),
                features=np.array(feat), thresholds=np.array(th), values=np.array(val).reshape(-1, 1),
                node_sample_weight=np.maximum(np.array(cnt), 1e-3))      # TreeSHAP needs a positive cover
    leaf_val = tree["values"][:, 0]
    return tree, [leaf_val[n] for n in nid]


def tree_apply(tree, X):
    node = np.zeros(len(X), int)
    for _ in range(DEPTH + 1):
        f = tree["features"][node]; internal = f >= 0
        go_left = X[np.arange(len(X)), np.where(internal, f, 0)] <= tree["thresholds"][node]
        node = np.where(internal, np.where(go_left, tree["children_left"][node], tree["children_right"][node]), node)
    return node


class Ensemble:
    def __init__(self, trees=None): self.trees = trees or []
    def margin(self, X):
        out = np.zeros(len(X))
        for t in self.trees:
            out += t["values"][tree_apply(t, X), 0]
        return out
    def explainer(self):
        return shap.TreeExplainer({"trees": [dict(t) for t in self.trees], "base_offset": 0.0, "tree_output": "raw_value",
                                   "objective": "binary_logistic", "input_dtype": np.float64, "internal_dtype": np.float64})
    def shap(self, X):
        return np.asarray(self.explainer().shap_values(X, check_additivity=False))


class Model:
    """A global model: weighted sum of ensembles (one ensemble for hist/cyclic/pooled)."""
    def __init__(self, ens, w=None): self.ens = ens; self.w = w or [1.0] * len(ens)
    def margin(self, X): return sum(w * e.margin(X) for w, e in zip(self.w, self.ens))
    def prob(self, X): return sig(self.margin(X))
    def shap(self, X): return sum(w * e.shap(X) for w, e in zip(self.w, self.ens))


# ------------------------------------------------------------------ training procedures
def fit_hist(clients, thr, sigma=0.0, seed=0, ntree=NTREE):
    Xs = [c[0] for c in clients]; ys = [c[1] for c in clients]
    yall = np.concatenate(ys); n, n1 = len(yall), yall.sum()
    ws = [np.where(y == 1, n / (2 * n1), n / (2 * (n - n1))) for y in ys]
    Bm = [binned(X, thr) for X in Xs]
    marg = [np.zeros(len(y)) for y in ys]
    rng = np.random.default_rng(seed); ens = Ensemble()
    for _ in range(ntree):
        p = [sig(m) for m in marg]
        G = [w * (pi - y) for w, pi, y in zip(ws, p, ys)]; H = [w * pi * (1 - pi) for w, pi in zip(ws, p)]
        sd = sigma * sum(h.sum() for h in H)
        tree, upd = fit_tree(Bm, G, H, thr, sd, rng)
        ens.trees.append(tree)
        marg = [m + u for m, u in zip(marg, upd)]
    return Model([ens])


def fit_bagging(clients, thr):
    n = sum(len(c[1]) for c in clients)
    return Model([fit_hist([c], thr).ens[0] for c in clients], [len(c[1]) / n for c in clients])


def fit_cyclic(clients, thr, ntree=NTREE):
    ys = [c[1] for c in clients]; yall = np.concatenate(ys); n, n1 = len(yall), yall.sum()
    ws = [np.where(y == 1, n / (2 * n1), n / (2 * (n - n1))) for y in ys]
    Bm = [binned(c[0], thr) for c in clients]
    marg = [np.zeros(len(y)) for y in ys]; ens = Ensemble()
    for t in range(ntree):
        k = t % len(clients); p = sig(marg[k])
        tree, upd = fit_tree([Bm[k]], [ws[k] * (p - ys[k])], [ws[k] * p * (1 - p)], thr)
        ens.trees.append(tree)
        for i, c in enumerate(clients):                      # every client updates its own margin with the new tree
            marg[i] = marg[i] + tree["values"][tree_apply(tree, c[0]), 0]
    return Model([ens])


# ------------------------------------------------------------------ evaluation
def topm(A): return np.argsort(-np.abs(A), 1)[:, :M]


def agree(A, Aref):
    ta, tb = topm(A), topm(Aref)
    jac = np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)])
    cos = (A * Aref).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Aref, axis=1) + 1e-12)
    return dict(jac=jac.mean(), top1=(ta[:, 0] == tb[:, 0]).mean(), cos=cos.mean(),
                exact=np.mean([set(a) == set(b) for a, b in zip(ta, tb)]))


def run_dataset(name, Xtr, ytr, Xte, yte, part_fn, regimes):
    thr = thresholds(Xtr)
    ref = fit_hist([(Xtr, ytr)], thr)
    T0 = float(np.quantile(ref.margin(Xtr), 0.70))
    auc_ref = roc_auc_score(yte, ref.margin(Xte))
    boots = []
    for b in range(B):
        r = np.random.default_rng(10_000 + b); idx = r.integers(0, len(ytr), len(ytr))
        boots.append(fit_hist([(Xtr[idx], ytr[idx])], thr))
    Xdec = Xtr[ref.margin(Xtr) >= T0]
    Aref = ref.shap(Xdec)
    fl = pd.DataFrame([agree(bm.shap(Xdec), Aref) for bm in boots])
    floor = dict(dataset=name, jac_mean=fl.jac.mean(), jac_p05=fl.jac.quantile(0.05), top1_mean=fl.top1.mean(),
                 top1_p05=fl.top1.quantile(0.05), cos_mean=fl.cos.mean(), cos_p05=fl.cos.quantile(0.05), exact_mean=fl.exact.mean(),
                 n_declined=len(Xdec), ref_test_auc=auc_ref)
    print(f"[{time.time()-t0:.0f}s] {name}: reference AUC {auc_ref:.3f}, floor jac {floor['jac_mean']:.3f} ({floor['jac_p05']:.3f}), declined {len(Xdec)}", flush=True)
    rows = []
    for regime in regimes:
        for seed in SEEDS:
            cl = part_fn(regime, seed)
            declined = [(Xk[ref.margin(Xk) >= T0]) for Xk, _, _ in cl]
            Xd = np.concatenate(declined); Ar = ref.shap(Xd)
            models = {"hist": fit_hist(cl, thr), "bagging": fit_bagging(cl, thr), "cyclic": fit_cyclic(cl, thr)}
            if name == "bangladesh" and regime == "high":
                for s in SIGMAS:
                    models[f"hist_noise_{s}"] = fit_hist(cl, thr, sigma=s, seed=seed)
            iso = [fit_hist([(Xk, yk)], thr) for Xk, yk, _ in cl]
            for mname, m in models.items():
                r = agree(m.shap(Xd), Ar)
                rows.append(dict(dataset=name, regime=regime, seed=seed, method=mname, clients=len(cl),
                                 test_auc=roc_auc_score(yte, m.margin(Xte)), n_declined=len(Xd), **r))
            Ai = np.concatenate([iso[k].shap(declined[k]) for k in range(len(cl)) if len(declined[k])])
            Ar_i = np.concatenate([ref.shap(declined[k]) for k in range(len(cl)) if len(declined[k])])
            rows.append(dict(dataset=name, regime=regime, seed=seed, method="isolated", clients=len(cl),
                             test_auc=float(np.mean([roc_auc_score(yte, p.margin(Xte)) for p in iso])), n_declined=len(Ai), **agree(Ai, Ar_i)))
        print(f"[{time.time()-t0:.0f}s] {name} {regime} done", flush=True)
    return rows, floor


def main():
    names = sys.argv[1:] or ["bangladesh", "german", "taiwan_default", "taiwan_bankruptcy", "lending_club"]
    R, F = [], []
    if len(sys.argv) > 1:                                   # adding datasets: keep earlier rows
        for f_, L in [("gbdt_fl_vs_reference", R), ("gbdt_noise_floor", F)]:
            try: L += [r for r in pd.read_csv(f"{OUT}/{f_}.csv").to_dict("records") if r["dataset"] not in names]
            except FileNotFoundError: pass
    for nm in names:
        if nm == "bangladesh":
            Xtr, ytr, Xte, yte = v2.Xtr, v2.ytr, v2.Xte, v2.yte
            pf = lambda regime, seed: ed.partition(regime, seed)
            regs = ["low", "medium", "high"]
        else:
            Xtr, ytr, Xte, yte, pref, *_ = xd.prepare(nm)
            pf = lambda regime, seed, pref=pref, Xtr=Xtr, ytr=ytr: xd.partition_pref(pref, Xtr, ytr, regime, np.random.default_rng(seed))
            regs = ["natural"] if nm in xd.NATURAL else ["medium", "high"]
        r, f = run_dataset(nm, Xtr, ytr, Xte, yte, pf, regs); R += r; F.append(f)
        pd.DataFrame(R).to_csv(f"{OUT}/gbdt_fl_vs_reference.csv", index=False)
        pd.DataFrame(F).to_csv(f"{OUT}/gbdt_noise_floor.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
