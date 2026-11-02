#!/usr/bin/env python3
"""LaTeX tables for the manuscript, generated from the result CSVs so no number is typed by hand.
Run from this folder: python make_tables.py   (writes tables/*.tex)"""
import os
import numpy as np, pandas as pd

R = "results_drift"; V2 = "results_federated_v2"; V3 = "results_federated_v3"; OUT = "tables"
os.makedirs(OUT, exist_ok=True)
NAMES = {"centralised": "Pooled model (reference)", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
         "median": "Coordinate-wise median", "isolated": "Isolated local"}
ORDER = ["fedavg", "fedprox", "scaffold", "median", "isolated"]


def w(name, lines):
    lines = [l + "\\" if l.endswith("\\") and not l.endswith("\\\\") else l for l in lines]   # row terminators written as one backslash
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines) + "\n")


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"


# ---------------------------------------------------------------- Table: classifier comparison
z = pd.read_csv(f"{V2}/model_zoo.csv")
rows = ["\\begin{tabular}{lccc}", "\\toprule", "Classifier & ROC-AUC (mean $\\pm$ sd) & PR-AUC & Brier score \\\\", "\\midrule"]
for _, r in z.iterrows():
    nm = r.Model.replace("(1x32)", "(1$\\times$32)").replace("(k=15)", "($k$=15)").replace("k-NN", "$k$-NN")
    rows.append(f"{nm} & {r.ROC_AUC:.3f} $\\pm$ {r.ROC_AUC_std:.3f} & {r.PR_AUC:.3f} & {r.Brier:.3f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
w("zoo", rows)

# ---------------------------------------------------------------- Table: main drift results (high heterogeneity)
s1 = pd.read_csv(f"{R}/summary_e1.csv"); s1 = s1[(s1.baseline == "shared") & (s1.regime == "high")]
fl = pd.read_csv(f"{R}/summary_noise_floor.csv"); fl = fl[(fl.baseline == "shared") & (fl.floor == "boot")].set_index("model")
cv = pd.read_csv(f"{V2}/cv_auc_high.csv"); cv["aggregator"] = cv.aggregator.replace({"centralized": "centralised"})
rows = ["\\begin{tabular}{lcccccl}", "\\toprule",
        "Setting & \\makecell{CV\\\\ROC-AUC} & \\makecell{Top-5\\\\overlap} & \\makecell{Same top\\\\reason} & Cosine & \\makecell{Identical\\\\top-5 sets} & \\makecell{Below floor\\\\on} \\\\", "\\midrule"]
for m, title in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    f = fl.loc[m]
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
    a = cv[(cv.model == m) & (cv.aggregator == "centralised")].iloc[0].cv_auc
    rows.append(f"Pooled model (reference) & {a:.3f} & 1 & 1 & 1 & 1 & \\\\")
    for k in ORDER:
        r = s1[(s1.model == m) & (s1.method == k)].iloc[0]
        a = cv[(cv.model == m) & (cv.aggregator == k)].iloc[0].cv_auc
        below = []
        if r.jac < f.jac_p05: below.append("overlap")
        if r.top1 < f.top1_p05: below.append("top reason")
        if r.cos < f.cos_p05: below.append("cosine")
        rows.append(f"{NAMES[k]} & {a:.3f} & {r.jac:.3f} & {r.top1:.3f} & {r.cos:.3f} & {r.exact:.3f} & {', '.join(below) if below else 'none'} \\\\")
    rows.append(f"Retraining noise floor: mean (5th pct) & & {f.jac_mean:.3f} ({f.jac_p05:.3f}) & {f.top1_mean:.3f} ({f.top1_p05:.3f}) & {f.cos_mean:.3f} ({f.cos_p05:.3f}) & {f.exact_mean:.3f} & \\\\")
    if m == "LR":
        rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("drift_high", rows)

# ---------------------------------------------------------------- Table: vertical and personalised
v = pd.read_csv(f"{V3}/vfl_fragmentation.csv").set_index("model")
p = pd.read_csv(f"{V3}/pfl_tension.csv").set_index("model")
fl_all = pd.read_csv(f"{R}/summary_noise_floor.csv"); fl_all = fl_all[(fl_all.baseline == "shared") & (fl_all.floor == "boot")].set_index("model")
def _pfl_cv():
    blk = open(f"{V2}/cv_extra_log.txt").read().split("== pfl ==")[1].split("==")[0].strip().splitlines()
    out = {}
    for ln in blk:
        t = ln.split()          # e.g. "LR global 0.8946 pers 0.8938"
        out[t[0]] = (float(t[2]), float(t[4]))
    return out
cvx = _pfl_cv()   # parsed from cv_extra_log.txt, pfl block
rows = ["\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{3}{c}{Vertical federation} & \\multicolumn{3}{c}{Personalised federation} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Model & \\makecell{Parties\\\\spanned} & \\makecell{Best single\\\\party} & \\makecell{Party-local\\\\$\\rho$} & \\makecell{Local fidelity\\\\(global $\\to$ pers.)} & \\makecell{Top-5 sets differ\\\\(global $\\to$ pers.)} & \\makecell{CV ROC-AUC\\\\(global $\\to$ pers.)} \\\\", "\\midrule"]
for m, nm in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    rows.append(f"{nm} & {v.loc[m].reasons_per_instance:.2f} & {v.loc[m].single_party_coverage:.3f} & {v.loc[m].party_local_vs_global_rho:.3f} & "
                f"{p.loc[m].local_fidelity_global:.3f} $\\to$ {p.loc[m].local_fidelity_personalized:.3f} & "
                f"{p.loc[m].crossclient_viol_global:.3f} $\\to$ {p.loc[m].crossclient_viol_personalized:.3f} & {cvx[m][0]:.3f} $\\to$ {cvx[m][1]:.3f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
w("struct", rows)

# ---------------------------------------------------------------- Table: training cost
c = pd.read_csv(f"{R}/e4_cost_training.csv")
rows = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Setting & \\makecell{Row-gradient\\\\evaluations} & \\makecell{Parameters\\\\per message} & \\makecell{Floats exchanged\\\\(20 rounds)} & \\makecell{Time\\\\(s)} \\\\", "\\midrule"]
for m, title in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    for k in ["centralised"] + ORDER:
        r = c[(c.model == m) & (c.method == k)].iloc[0]
        rows.append(f"{title if k=='centralised' else ''} & {NAMES[k].replace(' (reference)','')} & {int(r.row_gradients):,} & {int(r.parameters):,} & {int(r.floats_communicated):,} & {r.seconds:.2f} \\\\")
    if m == "LR": rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("cost_training", rows)

# ---------------------------------------------------------------- Table: explainers (cost, concentration, agreement)
ce = pd.read_csv(f"{R}/e4_cost_explanation.csv"); ag = pd.read_csv(f"{R}/e5_explainer_agreement.csv")
co = pd.read_csv(f"{R}/e5_concentration.csv"); co = co[co.source == "reference"]
def fmt_t(t):
    if t < 1e-3: return f"{t*1e6:.0f} $\\mu$s"
    if t < 1: return f"{t*1e3:.1f} ms"
    return f"{t:.1f} s"
rows = ["\\begin{tabular}{llcccccc}", "\\toprule",
        "Model & Explainer & \\makecell{Time per 100\\\\applicants} & \\makecell{Top-5\\\\share} & Gini & \\makecell{Effective\\\\features} & \\makecell{Top-5 overlap\\\\with native} & \\makecell{Same top\\\\reason} \\\\", "\\midrule"]
for m, title in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    sub = ce[ce.model == m]
    first = True
    for _, r in sub.iterrows():
        cc = co[(co.model == m) & (co.explainer == r.explainer)].iloc[0]; a = ag[(ag.model == m) & (ag.explainer == r.explainer)].iloc[0]
        nm = r.explainer.replace("(2,136 samples, 20 background rows)", "").replace("(5,000 samples)", "").replace("(12 steps)", "").strip()
        rows.append(f"{title if first else ''} & {nm} & {fmt_t(r.seconds_per_100)} & {cc.top5_share:.3f} & {cc.gini:.3f} & {cc.effective_features:.1f} & "
                    f"{a.top5_jaccard_vs_native:.3f} & {a.top1_match_vs_native:.2f} \\\\")
        first = False
    if m == "LR":
        t = ce[ce.model == "GB (reference)"].iloc[0]
        rows.append(f" & TreeSHAP (gradient boosting, for reference) & {fmt_t(t.seconds_per_100)} & & & & & \\\\")
        rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("explainers", rows)

# ---------------------------------------------------------------- Table: joint view (SAFE-style scores plus drift)
J = pd.read_csv(f"{R}/e6_joint.csv"); SF = pd.read_csv(f"{R}/e8_safe.csv")
rows = ["\\begin{tabular}{llcccccc}", "\\toprule",
        "Model & Setting & \\makecell{CV\\\\ROC-AUC} & \\makecell{Top-5 overlap\\\\with reference} & \\makecell{Explanation\\\\perturbation cosine} & AURGR & AURGE & \\makecell{Compliance\\\\score} \\\\", "\\midrule"]
for m, title in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    for k in ["centralised"] + ORDER:
        r = J[(J.model == m) & (J.source == k)].iloc[0]; q = SF[(SF.model == m) & (SF.source == k)].iloc[0]
        rows.append(f"{title if k=='centralised' else ''} & {NAMES[k].replace(' (reference)','')} & {r.cv_auc:.3f} & {r.drift_top5_overlap:.3f} & {r.perturbation_cosine:.3f} & {q.aurgr:.3f} & {q.aurge:.3f} & {q.score_arith:.3f} \\\\")
    if m == "LR": rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("joint", rows)


# ---------------------------------------------------------------- Table: held-out group gaps (pooled vs FedAvg)
fz = pd.read_csv(f"{R}/e7_fairness.csv")
AT = {"sector": "Sector", "division": "Division", "location_type": "Location type", "owner_education": "Owner education"}
rows = ["\\begin{tabular}{llcccc}", "\\toprule",
        "Model & Attribute & \\makecell{Groups\\\\(loans per group)} & DP gap & EO gap & PE gap \\\\", "\\midrule"]
for m, title in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    for i, a in enumerate(AT):
        p = fz[(fz.model == m) & (fz.setting == "pooled") & (fz.attribute == a)].iloc[0]
        f = fz[(fz.model == m) & (fz.setting == "fedavg") & (fz.attribute == a)].iloc[0]
        rows.append(f"{title if i == 0 else ''} & {AT[a]} & {int(p.groups)} ({int(p.min_group)}--{int(p.max_group)}) & "
                    f"{p.DP_gap:.3f} $\\to$ {f.DP_gap:.3f} & {p.EO_gap:.3f} $\\to$ {f.EO_gap:.3f} & {p.PE_gap:.3f} $\\to$ {f.PE_gap:.3f} \\\\")
    if m == "LR": rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("fairness", rows)

# ---------------------------------------------------------------- Table: three public credit datasets
ER = pd.read_csv(f"{R}/ext_fl_vs_reference.csv"); EF = pd.read_csv(f"{R}/ext_noise_floor.csv"); EI = pd.read_csv(f"{R}/ext_dataset_info.csv")
DN = {"german": "German credit", "taiwan_default": "Taiwan card default", "taiwan_bankruptcy": "Taiwan bankruptcy"}
rows = ["\\begin{tabular}{llccccccc}", "\\toprule",
        "Dataset & Model & \\makecell{Pooled\\\\test AUC} & \\makecell{Noise floor:\\\\mean (5th pct)} & FedAvg & FedProx & SCAFFOLD & Median & Isolated \\\\", "\\midrule"]
for di, dsn in enumerate(DN):
    for m, title in [("LR", "LR"), ("MLP", "NN")]:
        g = ER[(ER.dataset == dsn) & (ER.model == m) & (ER.regime == "high")]
        fl = EF[(EF.dataset == dsn) & (EF.model == m) & (EF.regime == "high")]
        fm, fp = fl.jac_mean.mean(), fl.jac_p05.mean(); info = EI[(EI.dataset == dsn) & (EI.model == m)].iloc[0]
        cells = []
        for k in ["fedavg", "fedprox", "scaffold", "median", "isolated"]:
            v = g[g.method == k].jac.mean(); cells.append(f"{v:.3f}" + ("$^\\dagger$" if v < fp else ""))
        rows.append(f"{DN[dsn] if m == 'LR' else ''} & {title} & {info.ref_test_auc:.3f} & {fm:.3f} ({fp:.3f}) & " + " & ".join(cells) + " \\\\")
    if di < 2: rows.append("\\midrule")
rows += ["\\bottomrule", "\\end{tabular}"]
w("external", rows)

# ---------------------------------------------------------------- Table: concentration of FL sources (for text reference)
cs = pd.read_csv(f"{R}/e5_concentration.csv"); cs = cs[cs.source != "reference"]
cs.to_csv(f"{OUT}/concentration_sources.csv", index=False)

# ---------------------------------------------------------------- Tables: boosted trees, shared background, DP
GB = pd.read_csv(f"{R}/gbdt_fl_vs_reference.csv"); GF = pd.read_csv(f"{R}/gbdt_noise_floor.csv").set_index("dataset")
GN = {"hist": "Histogram aggregation (federated)", "bagging": "Per-client ensembles, averaged", "cyclic": "Cyclic (round robin)",
      "hist_noise_0.005": r"Histogram aggregation, noise $\sigma=0.005$", "hist_noise_0.02": r"Histogram aggregation, noise $\sigma=0.02$",
      "isolated": "Isolated local"}
rows = [r"\begin{tabular}{lccccl}", r"\toprule", r"Setting & \makecell{Test\\ROC-AUC} & \makecell{Top-5\\overlap} & \makecell{Same top\\reason} & Cosine & \makecell{Below floor\\on} \\", r"\midrule"]
fb = GF.loc["bangladesh"]
rows.append(f"Pooled boosted trees (reference) & {fb.ref_test_auc:.3f} & 1 & 1 & 1 & \\")
for mth in ["hist", "bagging", "cyclic", "hist_noise_0.005", "hist_noise_0.02", "isolated"]:
    g = GB[(GB.dataset == "bangladesh") & (GB.regime == "high") & (GB.method == mth)]
    below = [n for n, c, t in [("overlap", "jac", "jac_p05"), ("top reason", "top1", "top1_p05"), ("cosine", "cos", "cos_p05")] if g[c].mean() < fb[t]]
    rows.append(f"{GN[mth]} & {g.test_auc.mean():.3f} & {g.jac.mean():.3f} & {g.top1.mean():.3f} & {g.cos.mean():.3f} & {', '.join(below) if below else 'none'} \\")
rows.append(f"Retraining noise floor: mean (5th pct) & & {fb.jac_mean:.3f} ({fb.jac_p05:.3f}) & {fb.top1_mean:.3f} ({fb.top1_p05:.3f}) & {fb.cos_mean:.3f} ({fb.cos_p05:.3f}) & \\")
rows += [r"\bottomrule", r"\end{tabular}"]
w("gbdt", rows)

DS = [("bangladesh", "Bangladeshi SME loans"), ("german", "German credit"), ("taiwan_default", "Taiwan card default"), ("taiwan_bankruptcy", "Taiwan bankruptcy")]
rows = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Dataset & \makecell{Pooled\\test AUC} & \makecell{Noise floor:\\mean (5th pct)} & Histogram & Bagging & Cyclic & Isolated \\", r"\midrule"]
for dn, nm in DS:
    f = GF.loc[dn]; cells = []
    for mth in ["hist", "bagging", "cyclic", "isolated"]:
        v = GB[(GB.dataset == dn) & (GB.regime == "high") & (GB.method == mth)].jac.mean()
        cells.append(f"{v:.3f}" + (r"$^\dagger$" if v < f.jac_p05 else ""))
    rows.append(f"{nm} & {f.ref_test_auc:.3f} & {f.jac_mean:.3f} ({f.jac_p05:.3f}) & " + " & ".join(cells) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
w("gbdt_all", rows)

SB = pd.read_csv(f"{R}/sbg_fl_vs_reference.csv"); SF = pd.read_csv(f"{R}/sbg_noise_floor.csv").set_index("model")
rows = [r"\begin{tabular}{llcccccc}", r"\toprule", r"Model & Regime & \makecell{Noise floor:\\mean (5th pct)} & \makecell{Same model,\\other background} & \makecell{Pooled\\background} & \makecell{Local\\backgrounds} & \makecell{Shared federated\\background} \\", r"\midrule"]
for m, mn in [("LR", "Logistic regression"), ("MLP", "Neural network")]:
    for k, reg in enumerate(["medium", "high"]):
        f = SF.loc[m]; cells = []
        for st in ["fed_pooledbg", "fed_localbg", "fed_sharedbg"]:
            g = SB[(SB.model == m) & (SB.regime == reg) & (SB.setting == st)].overlap
            cells.append(f"{g.mean():.3f}" + (f" [{(g < f.floor_p05).sum()}/5]" if (g < f.floor_p05).sum() else ""))
        rows.append(f"{mn if k == 0 else ''} & {reg} & {f.floor_mean:.3f} ({f.floor_p05:.3f}) & {f.bg_noise_only:.3f} & " + " & ".join(cells) + r" \\")
    if m == "LR": rows.append(r"\midrule")
rows += [r"\bottomrule", r"\end{tabular}"]
w("sbg", rows)

DP = pd.read_csv(f"{R}/dp_fedavg.csv")
rows = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Setting & \makecell{Noise\\multiplier $z$} & \makecell{Test\\ROC-AUC} & \makecell{Top-5\\overlap} & sd & \makecell{Partitions\\below floor} \\", r"\midrule"]
fp = DP.floor_p05.iloc[0]; fm = DP.floor_mean.iloc[0]
for lab, nm in [("non-private", "FedAvg, non-private"), ("clip only", "Clipping only ($C=2$, no noise)")] + [(f"eps={e}", rf"$\varepsilon={e}$") for e in [32, 16, 8, 4, 2, 1]]:
    g = DP[(DP.regime == "high") & (DP.setting == lab)]; z = g.z.iloc[0]
    rows.append(f"{nm} & {'' if z == 0 else f'{z:.1f}'} & {g.test_auc.mean():.3f} & {g.overlap.mean():.3f} & {g.overlap.std():.3f} & {(g.overlap < fp).sum()}/5 \\")
rows.append(f"Retraining noise floor: mean (5th pct) & & & {fm:.3f} ({fp:.3f}) & & \\")
rows += [r"\bottomrule", r"\end{tabular}"]
w("dp", rows)

print("tables written:", sorted(os.listdir(OUT)))
