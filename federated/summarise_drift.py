#!/usr/bin/env python3
"""Aggregate the raw explanation_drift.py outputs into the summary tables used in the paper.
Reads results_drift/e1_*.csv, e2_*.csv, e3_*.csv; writes results_drift/summary_*.csv."""
import pandas as pd, numpy as np

R = "results_drift"
e1 = pd.read_csv(f"{R}/e1_fl_vs_reference.csv")
fl = pd.read_csv(f"{R}/e1_noise_floor.csv")
e2 = pd.read_csv(f"{R}/e2_rank_agreement_sources.csv")
e3 = pd.read_csv(f"{R}/e3_stress.csv")


def ms(g, cols):
    out = g[cols].agg(["mean", "std"]).T
    return out


# ---- E1: agreement with the centralised reference, shared baseline (primary) and client baselines
rows = []
for (model, regime, method, base), g in e1.groupby(["model", "regime", "method", "baseline"]):
    r = dict(model=model, regime=regime, method=method, baseline=base, n_seeds=len(g), n_declined=g.n_declined.mean(),
             test_auc=g.test_auc.mean())
    for c in ["cos", "jac", "top1", "exact"]:
        r[c] = g[c].mean(); r[c + "_sd"] = g[c].std()
    r["legacy_violation"] = 1 - g.exact.mean()
    rows.append(r)
s1 = pd.DataFrame(rows)

# noise floor (same for every partition seed because the declined set is the union of the clients' rows)
f = fl.groupby(["model", "baseline", "floor"]).agg(jac_mean=("jac_mean", "mean"), jac_p05=("jac_p05", "mean"),
                                                    cos_mean=("cos_mean", "mean"), cos_p05=("cos_p05", "mean"),
                                                    top1_mean=("top1_mean", "mean"), top1_p05=("top1_p05", "mean"),
                                                    exact_mean=("exact_mean", "mean")).reset_index()
s1 = s1.merge(f[f.floor == "boot"][["model", "baseline", "jac_mean", "jac_p05", "top1_mean", "top1_p05"]]
              .rename(columns={"jac_mean": "floor_jac", "jac_p05": "floor_jac_p05",
                               "top1_mean": "floor_top1", "top1_p05": "floor_top1_p05"}),
              on=["model", "baseline"])
s1["within_floor_jac"] = s1.jac >= s1.floor_jac_p05
# share of partition seeds whose own agreement falls below the floor's 5th percentile
_m = e1.merge(f[f.floor == "boot"][["model", "baseline", "jac_p05"]], on=["model", "baseline"])
_m["below"] = _m.jac < _m.jac_p05
s1 = s1.merge(_m.groupby(["model", "regime", "method", "baseline"]).below.mean().rename("share_seeds_below_floor").reset_index(),
              on=["model", "regime", "method", "baseline"])
s1.to_csv(f"{R}/summary_e1.csv", index=False)
f.to_csv(f"{R}/summary_noise_floor.csv", index=False)

# ---- E2
s2 = e2.groupby(["model", "regime"]).agg(["mean", "std"]).drop(columns="seed")
s2.columns = ["_".join(c) for c in s2.columns]
s2.reset_index().to_csv(f"{R}/summary_e2.csv", index=False)

# ---- E3
s3 = e3.groupby(["model", "factor", "level"]).agg(K=("K", "mean"), test_auc=("test_auc", "mean"), cos=("cos", "mean"),
                                                  jac=("jac", "mean"), jac_sd=("jac", "std"), top1=("top1", "mean"),
                                                  exact=("exact", "mean"), floor_jac=("floor_jac", "mean"),
                                                  floor_jac_p05=("floor_jac_p05", "mean"),
                                                  n_declined=("n_declined", "mean")).reset_index()
s3["within_floor_jac"] = s3.jac >= s3.floor_jac_p05
_b = e3.assign(below=e3.jac < e3.floor_jac_p05).groupby(["model", "factor", "level"]).below.mean().rename("share_seeds_below_floor").reset_index()
s3 = s3.merge(_b, on=["model", "factor", "level"])
s3.to_csv(f"{R}/summary_e3.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(s1[(s1.baseline == "shared")][["model", "regime", "method", "jac", "jac_sd", "top1", "cos", "exact", "floor_jac",
                                         "floor_jac_p05", "within_floor_jac"]].round(3).to_string(index=False))
    print(f.round(3).to_string(index=False))
