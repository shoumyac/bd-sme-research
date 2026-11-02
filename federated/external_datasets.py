#!/usr/bin/env python3
"""
Explanation-drift audit on three public credit datasets (generalisation check).

For each dataset the same audit as in explanation_drift.py is run: a pooled-data reference model, FedAvg, FedProx,
SCAFFOLD, coordinate-wise median and isolated local models, for logistic regression and the one-hidden-layer
network, judged against a retraining noise floor (50 bootstrap retrainings of the pooled model), on the
applicants the reference declines, with a shared baseline. The decline rule is a fixed approval policy:
the 30% of training applicants with the highest reference score are declined (threshold set per model at the 70th
percentile of the reference's training scores), so that the declined share is the same for every dataset.

Datasets (put the three CSV files in ../data/external/, see docs/EXTERNAL_DATA.md for sources and licences):
  german_credit.csv        Statlog German Credit Data, 1,000 loans, bad = 1 (30%)        (Hofmann, 1994; UCI)
  taiwan_default.csv       Default of credit card clients, Taiwan, 30,000 clients, 5,000 sampled here  (Yeh and Lien, 2009; UCI)
  taiwan_bankruptcy.csv    Taiwanese bankruptcy prediction, 6,819 firms, bankrupt = 1 (3.2%)  (Liang et al., 2016; UCI)

Preprocessing is fitted on the training rows only (standardise numeric columns; numeric values clipped to
+/- 5 standard deviations; one-hot vocabulary from the schema). Clients are stylised, as for the Bangladesh data:
three institution archetypes defined from borrower or firm attributes (see PARTITIONS below).

Usage: python external_datasets.py            -> results_drift/ext_*.csv
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import federated_xai_experiment_v2 as v2
import explanation_drift as ed

OUT = ed.OUT
DATA = "../data/external"
DECLINE_SHARE = 0.30; B = ed.B; SEEDS = ed.SEEDS
t0 = time.time()


# ------------------------------------------------------------------ loaders
def load_german():
    df = pd.read_csv(f"{DATA}/german_credit.csv")
    y = (df["creditability"] == "bad").astype(int).values
    df = df.drop(columns=["creditability", "personal_status_and_sex", "foreign_worker"])   # protected attributes removed
    num = [c for c in df.columns if df[c].dtype != object and str(df[c].dtype) != "str"]
    cat = [c for c in df.columns if c not in num]
    return df, y, num, cat


def load_taiwan_default():
    df = pd.read_csv(f"{DATA}/taiwan_default.csv")
    df = df.rename(columns={"X1": "limit_bal", "X2": "sex", "X3": "education", "X4": "marriage", "X5": "age",
                            **{f"X{6+i}": f"pay_status_{i+1}" for i in range(6)},
                            **{f"X{12+i}": f"bill_amt_{i+1}" for i in range(6)},
                            **{f"X{18+i}": f"pay_amt_{i+1}" for i in range(6)}, "Y": "default"})
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    y = df["default"].astype(int).values
    df = df.drop(columns=["default", "sex"])                                               # protected attribute removed
    df["education"] = df["education"].clip(0, 4).astype(str)
    df["marriage"] = df["marriage"].clip(0, 3).astype(str)
    cat = ["education", "marriage"]; num = [c for c in df.columns if c not in cat]
    return df, y, num, cat


def load_taiwan_bankruptcy():
    df = pd.read_csv(f"{DATA}/taiwan_bankruptcy.csv")
    df.columns = [c.strip() for c in df.columns]
    y = df["Bankrupt?"].astype(int).values
    df = df.drop(columns=["Bankrupt?"])
    df = df.loc[:, df.nunique() > 1]                                                       # drop constant columns
    return df, y, list(df.columns), []


# ------------------------------------------------------------------ stylised institution archetypes
def pref_german(df):
    p = df["purpose"].values; a = df["age_in_years"].values
    goods = np.isin(p, ["radio/television", "furniture/equipment", "domestic appliances", "repairs", "others"])
    auto = np.isin(p, ["car (new)", "car (used)"])
    biz = np.isin(p, ["business", "education", "retraining"])
    return np.c_[1.0 * goods + 0.3 * (a >= 40), 1.0 * auto + 0.3 * (a < 30), 1.0 * biz]


def pref_taiwan_default(df):
    e = df["education"].values; a = df["age"].values
    return np.c_[1.0 * (e == "1") + 0.6 * (e == "2") * (a >= 30), 1.0 * (e == "3") + 0.5 * (a >= 45), 0.8 * (a < 30) + 0.5 * (e == "2") * (a < 30)]


def pref_bankruptcy(df):
    lev = df["Borrowing dependency"].values
    q1, q2 = np.quantile(lev, [1 / 3, 2 / 3])
    return np.c_[1.0 * (lev <= q1), 1.0 * ((lev > q1) & (lev <= q2)), 1.0 * (lev > q2)]


SPECS = {"german": (load_german, pref_german), "taiwan_default": (load_taiwan_default, pref_taiwan_default),
         "taiwan_bankruptcy": (load_taiwan_bankruptcy, pref_bankruptcy)}


def partition_pref(pref, X, y, regime, rng, K=3):
    n = len(y)
    if regime == "high":
        asg = pref.argmax(1); z = pref.sum(1) == 0; asg[z] = rng.integers(0, 3, z.sum())
    else:
        bias = {"low": 0.6, "medium": 0.8}[regime]
        pr = np.full((n, 3), (1 - bias) / 3); pr[np.arange(n), pref.argmax(1)] += bias; pr /= pr.sum(1, keepdims=True)
        asg = np.array([rng.choice(3, p=pr[i]) for i in range(n)])
    return [(X[asg == k], y[asg == k], None) for k in range(3) if (asg == k).sum() >= 15]


def prepare(name):
    loader, pref_fn = SPECS[name]
    df, y, num, cat = loader()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    cats = [sorted(df[c].astype(str).unique()) for c in cat]
    steps = [("num", StandardScaler(), num)]
    if cat:
        steps.append(("cat", OneHotEncoder(categories=cats, handle_unknown="ignore"), cat))
    ct = ColumnTransformer(steps)
    dfs = df.astype({c: str for c in cat}) if cat else df
    ct.fit(dfs.iloc[tr])
    X = ct.transform(dfs); X = X.toarray() if hasattr(X, "toarray") else X; X = np.asarray(X, float)
    X[:, :len(num)] = np.clip(X[:, :len(num)], -5, 5)
    pref = pref_fn(df)
    return X[tr], y[tr], X[te], y[te], pref[tr], len(num), df.iloc[tr].reset_index(drop=True)


def run(name):
    Xtr, ytr, Xte, yte, pref_tr, nnum, dtr = prepare(name)
    d = Xtr.shape[1]
    # point the shared module code at this dataset
    v2.d = d; ed.d = d; ed.Xtr = Xtr; ed.ytr = ytr; v2.Xte = Xte; v2.yte = yte; ed.SHARED = Xtr.mean(0)
    print(f"[{time.time()-t0:.0f}s] {name}: train {Xtr.shape}, test {Xte.shape}, bad rate {ytr.mean():.3f}, d={d}")
    rows, floors, info = [], [], []
    for mn, model in v2.MODELS.items():
        pref = ed.train_central(model, Xtr, ytr)
        boots = []
        for b in range(B):
            r = np.random.default_rng(10_000 + b); idx = r.integers(0, len(ytr), len(ytr))
            boots.append(ed.train_central(model, Xtr[idx], ytr[idx]))
        T0 = float(np.quantile(model.prob(pref, Xtr), 1 - DECLINE_SHARE))
        auc_ref = roc_auc_score(yte, model.prob(pref, Xte))
        info.append(dict(dataset=name, model=mn, n_train=len(ytr), n_test=len(yte), d=d, bad_rate=float(ytr.mean()), ref_test_auc=auc_ref, decline_threshold=T0))
        for regime in ["medium", "high"]:
            for seed in SEEDS:
                cl = partition_pref(pref_tr, Xtr, ytr, regime, np.random.default_rng(seed))
                ds = ed.declined_sets(model, pref, cl, T0)
                locs = [v2.local_train(model, model.init(), Xk, yk, 300, 0.5) for Xk, yk, _ in cl]
                fits = {a: v2.federated(model, cl, a) for a in ed.AGGS}
                for a in ed.AGGS:
                    r = ed.agree_fl(model, fits[a], pref, ds, "shared")
                    rows.append(dict(dataset=name, model=mn, regime=regime, seed=seed, method=a, clients=len(cl),
                                     test_auc=float(roc_auc_score(yte, model.prob(fits[a], Xte))), **r))
                r = ed.agree_fl(model, None, pref, ds, "shared", locs=locs)
                rows.append(dict(dataset=name, model=mn, regime=regime, seed=seed, method="isolated", clients=len(cl),
                                 test_auc=float(np.mean([roc_auc_score(yte, model.prob(p, Xte)) for p in locs])), **r))
                f = ed.noise_floor(model, pref, boots, ds, "shared")
                floors.append(dict(dataset=name, model=mn, regime=regime, seed=seed,
                                   jac_mean=f.jac.mean(), jac_p05=f.jac.quantile(0.05), cos_mean=f.cos.mean(), cos_p05=f.cos.quantile(0.05),
                                   top1_mean=f.top1.mean(), top1_p05=f.top1.quantile(0.05), exact_mean=f.exact.mean()))
            print(f"[{time.time()-t0:.0f}s] {name} {mn} {regime} done")
    return rows, floors, info


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPECS)
    R, F, I = [], [], []
    for n in names:
        r, f, i = run(n); R += r; F += f; I += i
    pd.DataFrame(R).to_csv(f"{OUT}/ext_fl_vs_reference.csv", index=False)
    pd.DataFrame(F).to_csv(f"{OUT}/ext_noise_floor.csv", index=False)
    pd.DataFrame(I).to_csv(f"{OUT}/ext_dataset_info.csv", index=False)
    print("saved")
