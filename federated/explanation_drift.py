#!/usr/bin/env python3
"""
Federated explanation-drift experiments (agreement with a pooled-data reference model).

Question: does federated training change the explanations a borrower receives,
relative to what centralised training on the pooled data would give, by more than
ordinary retraining noise? Evaluated only on applicants the reference model declines.

Stages:  python3 explanation_drift.py e1 | e2 | e3 | all
Outputs: results_drift/*.csv        (random_state / seeds fixed in the code)
Re-uses model, partition and federation code from federated_xai_experiment_v2.py.
"""
import os, sys, time, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
from federated_xai_experiment_v2 import d, H

OUT = "results_drift"; os.makedirs(OUT, exist_ok=True)
Xtr, ytr, META = v2.Xtr, v2.ytr, v2.meta_tr
MODELS = v2.MODELS
T_DEFAULT = 0.30     # decline threshold on the reference model's class-balanced score
M = 5                # number of reasons cited in an adverse-action notice
B = 50               # bootstrap replicates for the retraining noise floor
SEEDS = [0, 1, 2, 3, 4]
AGGS = ["fedavg", "fedprox", "scaffold", "median"]
SHARED = Xtr.mean(0)
t0 = time.time()


def init_seeded(model, seed):
    if model.name == "LR":
        return model.init()
    rng = np.random.default_rng(1000 + seed)
    return model.flat(rng.normal(0, np.sqrt(2 / d), (d, H)), np.zeros(H),
                      rng.normal(0, np.sqrt(2 / H), (H,)), 0.0)


def train_central(model, X, y, init=None):
    p0 = model.init() if init is None else init
    return v2.local_train(model, p0, X, y, 400, 0.5)


def topm(A, m=M):
    return np.argsort(-np.abs(A), axis=1)[:, :m]


def pair_metrics(model, pA, pB, X, base, m=M):
    """Per-instance agreement between the explanations of models A and B under one baseline."""
    A = model.attribution(pA, X, base); Bm = model.attribution(pB, X, base)
    cos = v2.cos_rows(A, Bm)
    ta, tb = topm(A, m), topm(Bm, m)
    jac = np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)])
    top1 = (ta[:, 0] == tb[:, 0]).astype(float)
    exact = np.array([set(a) == set(b) for a, b in zip(ta, tb)]).astype(float)
    return cos, jac, top1, exact


def declined_sets(model, pref, clients, T):
    """List of (client index, declined rows) according to the reference model."""
    out = []
    for k, (Xk, yk, _) in enumerate(clients):
        keep = model.prob(pref, Xk) >= T
        out.append((k, Xk[keep], Xk.mean(0)))
    return out


def agree_fl(model, pfl, pref, dsets, mode, locs=None):
    """Instance-weighted agreement between an FL model (or per-client isolated models) and the reference."""
    C, J, S1, EX = [], [], [], []
    for k, Xs, mu_k in dsets:
        if len(Xs) == 0:
            continue
        base = SHARED if mode == "shared" else mu_k
        p = pfl if locs is None else locs[k]
        c, j, s, e = pair_metrics(model, p, pref, Xs, base)
        C.append(c); J.append(j); S1.append(s); EX.append(e)
    C, J, S1, EX = map(np.concatenate, (C, J, S1, EX))
    return dict(cos=C.mean(), jac=J.mean(), top1=S1.mean(), exact=EX.mean(), n_declined=len(C))


def noise_floor(model, pref, boots, dsets, mode):
    """Agreement between the reference and bootstrap-retrained centralised models (sampling noise)."""
    vals = [agree_fl(model, pb, pref, dsets, mode) for pb in boots]
    df = pd.DataFrame(vals)
    return df


def make_boots(model):
    boots_same, boots_init = [], []
    for b in range(B):
        r = np.random.default_rng(10_000 + b)
        idx = r.integers(0, len(ytr), len(ytr))
        boots_same.append(train_central(model, Xtr[idx], ytr[idx]))
        boots_init.append(train_central(model, Xtr[idx], ytr[idx], init_seeded(model, b + 1)))
    return boots_same, boots_init


def partition(regime, seed, K=3):
    v2.RNG = np.random.default_rng(seed)
    return v2.partition(META, Xtr, ytr, regime, K=K)


def rank_vec(model, p, X, base):
    return np.abs(model.attribution(p, X, base)).mean(0)


def s_rank(model, p, clients, mode):
    rks = [rank_vec(model, p, Xk, SHARED if mode == "shared" else Xk.mean(0)) for Xk, _, _ in clients]
    sp = [spearmanr(a, b).correlation for a, b in itertools.combinations(rks, 2)]
    return float(np.mean(sp))


# ---------------------------------------------------------------- E1
def stage_e1():
    rows, floors = [], []
    for mn, model in MODELS.items():
        pref = train_central(model, Xtr, ytr)
        boots_same, boots_init = make_boots(model)
        print(f"[{time.time()-t0:.0f}s] {mn}: reference and {B}x2 bootstrap models trained")
        for regime in ["low", "medium", "high"]:
            for seed in SEEDS:
                cl = partition(regime, seed)
                ds = declined_sets(model, pref, cl, T_DEFAULT)
                locs = [v2.local_train(model, model.init(), Xk, yk, 300, 0.5) for Xk, yk, _ in cl]
                fits = {a: v2.federated(model, cl, a) for a in AGGS}
                for mode in ["shared", "client"]:
                    for a in AGGS:
                        r = agree_fl(model, fits[a], pref, ds, mode)
                        rows.append(dict(model=mn, regime=regime, seed=seed, method=a, baseline=mode,
                                         test_auc=v2.aucp(model, fits[a]), **r))
                    r = agree_fl(model, None, pref, ds, mode, locs=locs)
                    rows.append(dict(model=mn, regime=regime, seed=seed, method="isolated", baseline=mode,
                                     test_auc=float(np.mean([v2.aucp(model, p) for p in locs])), **r))
                    for tag, bs in [("boot", boots_same), ("boot_init", boots_init)]:
                        f = noise_floor(model, pref, bs, ds, mode)
                        floors.append(dict(model=mn, regime=regime, seed=seed, baseline=mode, floor=tag,
                                           jac_mean=f.jac.mean(), jac_p05=f.jac.quantile(0.05),
                                           cos_mean=f.cos.mean(), cos_p05=f.cos.quantile(0.05),
                                           top1_mean=f.top1.mean(), top1_p05=f.top1.quantile(0.05),
                                           exact_mean=f.exact.mean()))
                print(f"[{time.time()-t0:.0f}s] {mn} {regime} seed {seed} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/e1_fl_vs_reference.csv", index=False)
    pd.DataFrame(floors).to_csv(f"{OUT}/e1_noise_floor.csv", index=False)
    print("e1 saved")


# ---------------------------------------------------------------- E2
def stage_e2():
    """Cross-client rank agreement: where does it come from? (data heterogeneity vs aggregation vs baseline)"""
    rows = []
    for mn, model in MODELS.items():
        pref = train_central(model, Xtr, ytr)
        for regime in ["low", "medium", "high"]:
            for seed in SEEDS:
                cl = partition(regime, seed)
                pfa = v2.federated(model, cl, "fedavg")
                rows.append(dict(model=mn, regime=regime, seed=seed,
                                 centralised_client_baseline=s_rank(model, pref, cl, "client"),
                                 fedavg_client_baseline=s_rank(model, pfa, cl, "client"),
                                 centralised_shared_baseline=s_rank(model, pref, cl, "shared"),
                                 fedavg_shared_baseline=s_rank(model, pfa, cl, "shared")))
        print(f"[{time.time()-t0:.0f}s] e2 {mn} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/e2_rank_agreement_sources.csv", index=False)
    print("e2 saved")


# ---------------------------------------------------------------- E3
def stage_e3():
    rows = []
    for mn, model in MODELS.items():
        pref = train_central(model, Xtr, ytr)
        boots_same, boots_init = make_boots(model)
        # (a) severe label skew
        for alpha in [0.1, 0.5, 5.0]:
            for seed in SEEDS:
                v2.RNG = np.random.default_rng(seed)
                cl = v2.partition_dir(Xtr, ytr, alpha)
                ds = declined_sets(model, pref, cl, T_DEFAULT)
                pf = v2.federated(model, cl, "fedavg")
                r = agree_fl(model, pf, pref, ds, "shared")
                fl = noise_floor(model, pref, boots_same, ds, "shared")
                rows.append(dict(model=mn, factor="dirichlet_alpha", level=alpha, seed=seed, K=len(cl),
                                 test_auc=v2.aucp(model, pf), floor_jac=fl.jac.mean(), floor_jac_p05=fl.jac.quantile(0.05), **r))
        # (b) number of clients
        for K in [3, 5, 10]:
            for seed in SEEDS:
                cl = partition("high", seed, K=K)
                ds = declined_sets(model, pref, cl, T_DEFAULT)
                pf = v2.federated(model, cl, "fedavg")
                r = agree_fl(model, pf, pref, ds, "shared")
                fl = noise_floor(model, pref, boots_same, ds, "shared")
                rows.append(dict(model=mn, factor="clients", level=K, seed=seed, K=len(cl),
                                 test_auc=v2.aucp(model, pf), floor_jac=fl.jac.mean(), floor_jac_p05=fl.jac.quantile(0.05), **r))
        # (c) differential-privacy noise on the aggregate (LR only, as in the original ablation)
        if mn == "LR":
            for sigma in [0.0, 0.01, 0.05, 0.1, 0.2]:
                for seed in SEEDS:
                    cl = partition("high", seed)
                    ds = declined_sets(model, pref, cl, T_DEFAULT)
                    rng = np.random.default_rng(500 + seed)
                    p = model.init()
                    w = np.array([len(c[1]) for c in cl], float); w /= w.sum()
                    for t in range(20):
                        ps = [v2.local_train(model, p, Xk, yk, 18, 0.5) for Xk, yk, _ in cl]
                        p = np.sum([wi * pi for wi, pi in zip(w, ps)], 0)
                        if sigma > 0:
                            p = p + rng.normal(0, sigma, p.shape)
                    r = agree_fl(model, p, pref, ds, "shared")
                    fl = noise_floor(model, pref, boots_same, ds, "shared")
                    rows.append(dict(model=mn, factor="dp_sigma", level=sigma, seed=seed, K=len(cl),
                                     test_auc=v2.aucp(model, p), floor_jac=fl.jac.mean(), floor_jac_p05=fl.jac.quantile(0.05), **r))
        # (d) threshold sensitivity for the declined set (FedAvg, high heterogeneity)
        for T in [0.2, 0.3, 0.4, 0.5]:
            for seed in SEEDS:
                cl = partition("high", seed)
                ds = declined_sets(model, pref, cl, T)
                pf = v2.federated(model, cl, "fedavg")
                r = agree_fl(model, pf, pref, ds, "shared")
                fl = noise_floor(model, pref, boots_same, ds, "shared")
                rows.append(dict(model=mn, factor="threshold", level=T, seed=seed, K=len(cl),
                                 test_auc=v2.aucp(model, pf), floor_jac=fl.jac.mean(), floor_jac_p05=fl.jac.quantile(0.05), **r))
        print(f"[{time.time()-t0:.0f}s] e3 {mn} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/e3_stress.csv", index=False)
    print("e3 saved")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"e1": stage_e1, "e2": stage_e2, "e3": stage_e3}.get(st, lambda: [stage_e1(), stage_e2(), stage_e3()])()
    print(f"[{time.time()-t0:.0f}s] '{st}' complete")
