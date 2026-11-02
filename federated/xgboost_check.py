#!/usr/bin/env python3
"""
Established-library check of the federated boosted-tree results (XGBoost).

1. Validation: the pooled boosted trees of federated_gbdt.py (own implementation) against XGBoost trained on the pooled
   rows with matching settings (50 trees, depth 3, learning rate 0.1, lambda 1, min child hessian 1, 16 bins, class
   weights from the pooled class counts, base margin 0). XGBoost's own TreeSHAP (pred_contribs) explains it.
2. Federated variants with XGBoost itself, compared with the XGBoost pooled reference and its 50-bootstrap noise floor:
     hist      true horizontal histogram aggregation: one XGBoost worker per client (separate processes) connected by
               XGBoost's collective communicator; quantile sketches and gradient histograms are merged by the library
     bagging   every client trains its own XGBoost model, the global model averages margins and SHAP values by client size
     cyclic    one tree at a time, each on one client's rows (xgb_model continuation), round robin
     isolated  every client keeps its own model
   (No secure aggregation: the communicator is plain, which changes nothing in the numbers.)

Usage: python xgboost_check.py [dataset ...]   -> results_drift/xgb_*.csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import xgboost as xgb

PARAMS = dict(objective="binary:logistic", max_depth=3, eta=0.1, reg_lambda=1.0, min_child_weight=1.0, tree_method="hist",
              max_bin=16, base_score=0.5, nthread=1, seed=0, verbosity=0)
NTREE, B, M, SEEDS = 50, 50, 5, range(5)


def weights(y, n, n1):
    return np.where(y == 1, n / (2 * n1), n / (2 * (n - n1)))


def train(X, y, w, nround=NTREE, base=None):
    d = xgb.DMatrix(X, label=y, weight=w)
    return xgb.train(PARAMS, d, num_boost_round=nround, xgb_model=base)


def margin(b, X): return b.predict(xgb.DMatrix(X), output_margin=True)
def shapv(b, X): return b.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


class Avg:
    """Weighted average of boosters (margins and SHAP values)."""
    def __init__(self, bs, w): self.bs, self.w = bs, np.asarray(w)
    def margin(self, X): return sum(w * margin(b, X) for w, b in zip(self.w, self.bs))
    def shap(self, X): return sum(w * shapv(b, X) for w, b in zip(self.w, self.bs))


class One:
    def __init__(self, b): self.b = b
    def margin(self, X): return margin(self.b, X)
    def shap(self, X): return shapv(self.b, X)


# ------------------------------------------------------------------ distributed histogram aggregation
def _worker(args, X, y, w, nround):
    from xgboost import collective
    with collective.CommunicatorContext(**args):
        d = xgb.DMatrix(X, label=y, weight=w)
        b = xgb.train(PARAMS, d, num_boost_round=nround)
        return bytes(b.save_raw("ubj")) if collective.get_rank() == 0 else None


def fit_dist(pool, clients, ws):
    from xgboost.tracker import RabitTracker
    k = len(clients)
    tr = RabitTracker(n_workers=k, host_ip="127.0.0.1", sortby="task")
    tr.start()
    args = tr.worker_args()
    futs = [pool.submit(_worker, {**args, "dmlc_task_id": f"{i:08d}"}, c[0], c[1], w, NTREE) for i, (c, w) in enumerate(zip(clients, ws))]
    res = [f.result() for f in futs]
    tr.wait_for(60)
    b = xgb.Booster(); b.load_model(bytearray(next(r for r in res if r is not None)))
    return One(b)


# ------------------------------------------------------------------ other federated variants
def fit_bagging(clients, ws):
    n = sum(len(c[1]) for c in clients)
    return Avg([train(c[0], c[1], w) for c, w in zip(clients, ws)], [len(c[1]) / n for c in clients])


def fit_cyclic(clients, ws):
    b = None
    for t in range(NTREE):
        k = t % len(clients)
        b = train(clients[k][0], clients[k][1], ws[k], nround=1, base=b)
    return One(b)


def agree(A, Aref):
    ta, tb = np.argsort(-np.abs(A), 1)[:, :M], np.argsort(-np.abs(Aref), 1)[:, :M]
    jac = np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)])
    cos = (A * Aref).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Aref, axis=1) + 1e-12)
    return dict(jac=jac.mean(), top1=(ta[:, 0] == tb[:, 0]).mean(), cos=cos.mean())


def run_dataset(name, Xtr, ytr, Xte, yte, part_fn, regimes, pool, own):
    n, n1 = len(ytr), ytr.sum()
    ref = One(train(Xtr, ytr, weights(ytr, n, n1)))
    T0 = float(np.quantile(ref.margin(Xtr), 0.70))
    auc_ref = roc_auc_score(yte, ref.margin(Xte))
    Xdec = Xtr[ref.margin(Xtr) >= T0]; Aref = ref.shap(Xdec)
    fl = []
    for b in range(B):
        r = np.random.default_rng(10_000 + b); idx = r.integers(0, len(ytr), len(ytr))
        yb = ytr[idx]; bm = One(train(Xtr[idx], yb, weights(yb, n, yb.sum())))
        fl.append(agree(bm.shap(Xdec), Aref))
    fl = pd.DataFrame(fl)
    # validation: own implementation vs XGBoost, same pooled rows
    own_ref = own["fit_hist"]([(Xtr, ytr)], own["thr"])
    val = dict(dataset=name, xgb_test_auc=auc_ref, own_test_auc=roc_auc_score(yte, own_ref.margin(Xte)),
               margin_corr=float(np.corrcoef(own_ref.margin(Xtr), ref.margin(Xtr))[0, 1]),
               n_declined=len(Xdec), **{f"own_vs_xgb_{k}": v for k, v in agree(own_ref.shap(Xdec), Aref).items()})
    floor = dict(dataset=name, jac_mean=fl.jac.mean(), jac_p05=fl.jac.quantile(0.05), top1_mean=fl.top1.mean(),
                 top1_p05=fl.top1.quantile(0.05), cos_mean=fl.cos.mean(), cos_p05=fl.cos.quantile(0.05),
                 n_declined=len(Xdec), ref_test_auc=auc_ref)
    print(f"[{time.time()-t0:.0f}s] {name}: XGB AUC {auc_ref:.3f} (own {val['own_test_auc']:.3f}), own-vs-XGB overlap "
          f"{val['own_vs_xgb_jac']:.3f}, XGB floor {floor['jac_mean']:.3f} ({floor['jac_p05']:.3f})", flush=True)
    rows = []
    for regime in regimes:
        for seed in SEEDS:
            cl = part_fn(regime, seed)
            ws = [weights(c[1], n, n1) for c in cl]
            declined = [Xk[ref.margin(Xk) >= T0] for Xk, _, _ in cl]
            Xd = np.concatenate(declined); Ar = ref.shap(Xd)
            models = {"hist": fit_dist(pool, cl, ws), "bagging": fit_bagging(cl, ws), "cyclic": fit_cyclic(cl, ws)}
            for mname, m in models.items():
                rows.append(dict(dataset=name, regime=regime, seed=seed, method=mname, clients=len(cl),
                                 test_auc=roc_auc_score(yte, m.margin(Xte)), n_declined=len(Xd), **agree(m.shap(Xd), Ar)))
            iso = [One(train(c[0], c[1], w)) for c, w in zip(cl, ws)]
            ok = [k for k in range(len(cl)) if len(declined[k])]
            Ai = np.concatenate([iso[k].shap(declined[k]) for k in ok]); Ar_i = np.concatenate([ref.shap(declined[k]) for k in ok])
            rows.append(dict(dataset=name, regime=regime, seed=seed, method="isolated", clients=len(cl),
                             test_auc=float(np.mean([roc_auc_score(yte, p.margin(Xte)) for p in iso])), n_declined=len(Ai), **agree(Ai, Ar_i)))
        print(f"[{time.time()-t0:.0f}s] {name} {regime} done", flush=True)
    return rows, floor, val


def main():
    import federated_xai_experiment_v2 as v2
    import explanation_drift as ed
    import external_datasets as xd
    import federated_gbdt as g
    names = sys.argv[1:] or ["bangladesh", "german", "taiwan_default", "taiwan_bankruptcy", "lending_club"]
    pool = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"))
    R, F, V = [], [], []
    if len(sys.argv) > 1:                                   # adding datasets: keep earlier rows
        for f_, L in [("xgb_fl_vs_reference", R), ("xgb_noise_floor", F), ("xgb_validation", V)]:
            try: L += [r for r in pd.read_csv(f"{ed.OUT}/{f_}.csv").to_dict("records") if r["dataset"] not in names]
            except FileNotFoundError: pass
    for nm in names:
        if nm == "bangladesh":
            Xtr, ytr, Xte, yte = v2.Xtr, v2.ytr, v2.Xte, v2.yte
            pf = lambda regime, seed: ed.partition(regime, seed); regs = ["low", "medium", "high"]
        else:
            Xtr, ytr, Xte, yte, pref, *_ = xd.prepare(nm)
            pf = lambda regime, seed, pref=pref, Xtr=Xtr, ytr=ytr: xd.partition_pref(pref, Xtr, ytr, regime, np.random.default_rng(seed))
            regs = ["natural"] if nm in xd.NATURAL else ["medium", "high"]
        own = dict(fit_hist=g.fit_hist, thr=g.thresholds(Xtr))
        r, f, v = run_dataset(nm, Xtr, ytr, Xte, yte, pf, regs, pool, own); R += r; F.append(f); V.append(v)
        pd.DataFrame(R).to_csv(f"{ed.OUT}/xgb_fl_vs_reference.csv", index=False)
        pd.DataFrame(F).to_csv(f"{ed.OUT}/xgb_noise_floor.csv", index=False)
        pd.DataFrame(V).to_csv(f"{ed.OUT}/xgb_validation.csv", index=False)
    print("saved")


t0 = time.time()
if __name__ == "__main__":
    main()
