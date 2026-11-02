#!/usr/bin/env python3
"""
Computational cost, explanation concentration (parsimony), explainer agreement and a joint scorecard.

Run from this folder after explanation_drift.py (needs results_federated_v2/cv_auc_high.csv and
results_drift/summary_e1.csv):   python cost_concentration.py
Outputs (results_drift/):
  e4_cost_training.csv        wall-clock time, gradient work and communication volume per training method
  e4_cost_explanation.csv     time to explain 100 declined applicants per explanation method
  e5_concentration.csv        concentration of attributions (top-5 share, Gini, effective features, k80)
  e5_explainer_agreement.csv  agreement between explainers on the reference model
  e6_joint.csv                accuracy, drift, robustness and cost side by side, with a Pareto check
Requires shap and lime in addition to requirements.txt (pip install shap lime).
"""
import os, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
import explanation_drift as ed

OUT = ed.OUT
Xtr, ytr, SHARED, T, M = ed.Xtr, ed.ytr, ed.SHARED, ed.T_DEFAULT, ed.M
SEEDS = ed.SEEDS
AGGS = ["fedavg", "fedprox", "scaffold", "median"]
N_EXPLAIN = 100          # instances used for the explanation-cost and explainer-agreement runs
SIGMA = 0.1              # input-perturbation noise for the robustness check
REPS = 10
t0 = time.time()


def attrib_source(model, p, dsets, locs=None, noise=None):
    """Attributions on the declined sets (shared baseline); optional Gaussian input noise."""
    A = []
    for k, Xs, _ in dsets:
        if len(Xs) == 0:
            continue
        pk = p if locs is None else locs[k]
        X = Xs if noise is None else Xs + noise.normal(0, SIGMA, Xs.shape)
        A.append(model.attribution(pk, X, SHARED))
    return np.vstack(A)


def conc(A):
    """Concentration of |attribution| mass per instance, averaged: top-5 share, Gini, effective number, k80."""
    Pm = np.abs(A); Pm = Pm / Pm.sum(1, keepdims=True)
    srt = np.sort(Pm, axis=1)
    n = Pm.shape[1]
    top5 = srt[:, -5:].sum(1)
    gini = (2 * (srt * np.arange(1, n + 1)).sum(1)) / n - (n + 1) / n
    eff = 1.0 / (Pm ** 2).sum(1)
    k80 = (np.cumsum(srt[:, ::-1], axis=1) < 0.8).sum(1) + 1
    return dict(top5_share=top5.mean(), gini=gini.mean(), effective_features=eff.mean(), k80=k80.mean())


def timeit(fn, reps=5):
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    return float(np.median(ts))


# ---------------------------------------------------------------- E4a: training cost
def e4_training():
    rows = []
    for mn, model in ed.MODELS.items():
        cl = ed.partition("high", 0)
        P = len(model.init()); K = len(cl); n = sum(len(c[1]) for c in cl)
        rounds, epochs = 20, 18
        tc = timeit(lambda: ed.train_central(model, Xtr, ytr))
        rows.append(dict(model=mn, method="centralised", seconds=tc, row_gradients=400 * len(ytr),
                         floats_communicated=0, parameters=P, clients=1))
        for a in AGGS:
            ts = timeit(lambda a=a: v2.federated(model, cl, a))
            mult = 4 if a == "scaffold" else 2          # SCAFFOLD also sends control variates both ways
            rows.append(dict(model=mn, method=a, seconds=ts, row_gradients=rounds * epochs * n,
                             floats_communicated=rounds * K * mult * P, parameters=P, clients=K))
        tl = timeit(lambda: [v2.local_train(model, model.init(), Xk, yk, 300, 0.5) for Xk, yk, _ in cl])
        rows.append(dict(model=mn, method="isolated", seconds=tl, row_gradients=300 * n,
                         floats_communicated=0, parameters=P, clients=K))
        print(f"[{time.time()-t0:.0f}s] e4 training {mn}")
    df = pd.DataFrame(rows)
    df["kilobytes_float64"] = df.floats_communicated * 8 / 1024
    df.to_csv(f"{OUT}/e4_cost_training.csv", index=False)
    return df


# ---------------------------------------------------------------- E4b/E5: explainers
def explainers():
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(0)
    cost_rows, agree_rows, conc_rows = [], [], []
    bg = Xtr[rng.choice(len(Xtr), 20, replace=False)]
    for mn, model in ed.MODELS.items():
        pref = ed.train_central(model, Xtr, ytr)
        dec = Xtr[model.prob(pref, Xtr) >= T]
        X100 = dec[:N_EXPLAIN]
        f = lambda Z, p=pref, m=model: m.prob(p, Z)
        # native method: exact linear SHAP (LR) or integrated gradients (MLP)
        native = "linear SHAP (exact)" if mn == "LR" else "Integrated Gradients (12 steps)"
        t_nat = timeit(lambda: model.attribution(pref, X100, SHARED))
        A_nat = model.attribution(pref, X100, SHARED)
        # KernelSHAP (model-agnostic)
        ker = shap.KernelExplainer(f, bg)
        a = time.perf_counter(); A_ker = np.asarray(ker.shap_values(X100, nsamples=2136, silent=True)); t_ker = time.perf_counter() - a
        # LIME (all 44 coefficients, so that concentration and top-5 reasons can both be read)
        lime = LimeTabularExplainer(Xtr, mode="classification", discretize_continuous=False, random_state=0)
        pf = lambda Z: np.c_[1 - f(Z), f(Z)]
        a = time.perf_counter()
        A_lime = np.zeros_like(X100)
        for i, x in enumerate(X100):
            e = lime.explain_instance(x, pf, num_features=X100.shape[1], num_samples=5000)
            for j, w in e.as_map()[1]:
                A_lime[i, j] = w                       # LIME weight: local slope of the surrogate for feature j
        t_lime = time.perf_counter() - a
        # LIME weights are slopes; multiply by (x - baseline) so they are on the attribution scale
        A_lime = A_lime * (X100 - SHARED)
        for name, t, A in [(native, t_nat, A_nat), ("KernelSHAP (2,136 samples, 20 background rows)", t_ker, A_ker),
                           ("LIME (5,000 samples)", t_lime, A_lime)]:
            cost_rows.append(dict(model=mn, explainer=name, seconds_per_100=t, seconds_per_instance=t / len(X100)))
            conc_rows.append(dict(model=mn, source="reference", explainer=name, **conc(A)))
            ta, tb = ed.topm(A_nat), ed.topm(A)
            jac = np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)])
            top1 = np.mean(ta[:, 0] == tb[:, 0])
            rho = spearmanr(np.abs(A_nat).mean(0), np.abs(A).mean(0)).correlation
            agree_rows.append(dict(model=mn, explainer=name, top5_jaccard_vs_native=jac, top1_match_vs_native=top1,
                                   global_rank_spearman_vs_native=rho, instances=len(X100)))
        if mn == "LR":      # non-linear tree ensemble for cost reference (exact TreeSHAP)
            gb = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
            tx = shap.TreeExplainer(gb)
            tt = timeit(lambda: tx.shap_values(X100), reps=3)
            cost_rows.append(dict(model="GB (reference)", explainer="TreeSHAP (exact)", seconds_per_100=tt,
                                  seconds_per_instance=tt / len(X100)))
        print(f"[{time.time()-t0:.0f}s] explainers {mn}")
    pd.DataFrame(cost_rows).to_csv(f"{OUT}/e4_cost_explanation.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(f"{OUT}/e5_explainer_agreement.csv", index=False)
    return pd.DataFrame(conc_rows)


# ---------------------------------------------------------------- E5c / E6: sources
def sources():
    conc_rows, rob_rows = [], []
    for mn, model in ed.MODELS.items():
        pref = ed.train_central(model, Xtr, ytr)
        for seed in SEEDS:
            cl = ed.partition("high", seed)
            ds = ed.declined_sets(model, pref, cl, T)
            locs = [v2.local_train(model, model.init(), Xk, yk, 300, 0.5) for Xk, yk, _ in cl]
            fits = {"centralised": pref, **{a: v2.federated(model, cl, a) for a in AGGS}}
            for src, p in list(fits.items()) + [("isolated", None)]:
                A = attrib_source(model, p, ds, locs=locs if src == "isolated" else None)
                conc_rows.append(dict(model=mn, source=src, seed=seed, **conc(A)))
                rs = []
                for r in range(REPS):
                    nr = np.random.default_rng(100 * seed + r)
                    An = attrib_source(model, p, ds, locs=locs if src == "isolated" else None, noise=nr)
                    rs.append(v2.cos_rows(A, An).mean())
                rob_rows.append(dict(model=mn, source=src, seed=seed, perturbation_cosine=float(np.mean(rs))))
        print(f"[{time.time()-t0:.0f}s] sources {mn}")
    return pd.DataFrame(conc_rows), pd.DataFrame(rob_rows)


def main():
    train = e4_training()
    c_expl = explainers()
    c_src, rob = sources()
    c_src_m = c_src.groupby(["model", "source"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    c_src_m["explainer"] = np.where(c_src_m.model == "LR", "linear SHAP (exact)", "Integrated Gradients (12 steps)")
    pd.concat([c_expl, c_src_m], ignore_index=True).to_csv(f"{OUT}/e5_concentration.csv", index=False)
    rob_m = rob.groupby(["model", "source"]).agg(perturbation_cosine=("perturbation_cosine", "mean")).reset_index()

    # ---- joint scorecard (high heterogeneity, shared baseline)
    cv = pd.read_csv("results_federated_v2/cv_auc_high.csv").rename(columns={"aggregator": "source"})
    cv["source"] = cv.source.replace({"centralized": "centralised"})
    s1 = pd.read_csv(f"{OUT}/summary_e1.csv")
    s1 = s1[(s1.regime == "high") & (s1.baseline == "shared")][["model", "method", "jac"]].rename(columns={"method": "source", "jac": "drift_top5_overlap"})
    ref = pd.DataFrame([dict(model=m, source="centralised", drift_top5_overlap=1.0) for m in ["LR", "MLP"]])
    s1 = pd.concat([s1, ref], ignore_index=True)
    cost = train.rename(columns={"method": "source"})[["model", "source", "seconds", "floats_communicated"]]
    J = (cv[["model", "source", "cv_auc"]].merge(s1, on=["model", "source"]).merge(rob_m, on=["model", "source"])
         .merge(cost, on=["model", "source"]))
    # Pareto check among deployable settings (centralised pooling is excluded: it is what federation replaces).
    # A setting is dominated if another is at least as good on accuracy, drift agreement and communication
    # and strictly better on one.
    J["pareto_efficient"] = False
    for m in ["LR", "MLP"]:
        g = J[(J.model == m) & (J.source != "centralised")]
        for i, r in g.iterrows():
            dom = ((g.cv_auc >= r.cv_auc) & (g.drift_top5_overlap >= r.drift_top5_overlap) &
                   (g.floats_communicated <= r.floats_communicated) &
                   ((g.cv_auc > r.cv_auc) | (g.drift_top5_overlap > r.drift_top5_overlap) |
                    (g.floats_communicated < r.floats_communicated))).any()
            J.loc[i, "pareto_efficient"] = not dom
    J.to_csv(f"{OUT}/e6_joint.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] done")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(train.round(4).to_string(index=False))
    print(pd.read_csv(f"{OUT}/e4_cost_explanation.csv").round(4).to_string(index=False))
    print(pd.read_csv(f"{OUT}/e5_explainer_agreement.csv").round(3).to_string(index=False))
    print(pd.read_csv(f"{OUT}/e5_concentration.csv").round(3).to_string(index=False))
    print(J.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
