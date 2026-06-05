#!/usr/bin/env python3
"""
Federated Explainability Under Distribution Shift
=================================================
Federated simulation + cross-client SHAP stability analysis on the
1,200-record Bangladesh SME credit dataset.

Pipeline:
  1. Global preprocessing (one-hot + standardize) -> 44-dim feature space.
  2. Stratified 80/20 split: 80% federated pool, 20% global test.
  3. Partition the pool into 3 institution-type clients (MFI / Commercial /
     Cooperative) under low/medium/high heterogeneity regimes + Dirichlet skew.
  4. Train: Centralized, FedAvg, FedProx, Isolated-local logistic regression
     (custom NumPy LR so the FedProx proximal term is exact).
  5. Explanations: analytic linear SHAP  phi_j(x) = w_j * (x_j - mu_j)
     (exact SHAP for linear models, interventional baseline per client).
  6. Metrics: cross-client attribution stability S_cc, rank stability S_rank,
     top-5 Jaccard J5, top-m consistency violation rate, input-perturbation
     stability, per-client fairness gaps and subgroup AUC.
  7. Figures (a)-(d) and CSV outputs.

Reproducible: random_state = 42 everywhere.
Runs in seconds on CPU; Colab-ready.
"""
import numpy as np, pandas as pd, json, os, itertools, warnings
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
np.random.seed(42)
RNG = np.random.default_rng(42)

OUT = "results_federated"
os.makedirs(OUT, exist_ok=True)
DATA = "../data/bangladesh_sme_1200_records.csv"
if not os.path.exists(DATA):
    DATA = "data/bangladesh_sme_1200_records.csv"

# ----------------------------------------------------------------------
# 1. Load & preprocess
# ----------------------------------------------------------------------
df = pd.read_csv(DATA)
TARGET = "target_missed_payment"
NUM = ["business_age_years","employees_count","monthly_revenue_bdt","monthly_profit_bdt",
       "prior_loan_count","loan_amount_bdt","preferred_tenure_months","cashflow_stability",
       "repayment_confidence","days_payable_outstanding"]
CAT = ["sector","legal_status","division","location_type","existing_loans",
       "late_payment_history","loan_purpose","collateral_level","owner_education"]
y = df[TARGET].astype(int).values

ct = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
X = ct.fit_transform(df).astype(float)
if hasattr(X, "toarray"): X = X.toarray()
feat_names = NUM + list(ct.named_transformers_["cat"].get_feature_names_out(CAT))
d = X.shape[1]
print(f"Feature space: {X.shape}, {d} dimensions")

# keep raw subgroup columns aligned to rows for fairness audit
meta = df[["sector","division","location_type","owner_education"]].reset_index(drop=True)

idx = np.arange(len(y))
tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)
Xtr, ytr = X[tr_idx], y[tr_idx]
Xte, yte = X[te_idx], y[te_idx]
meta_tr = meta.iloc[tr_idx].reset_index(drop=True)
meta_te = meta.iloc[te_idx].reset_index(drop=True)
print(f"Train pool {len(ytr)}  Test {len(yte)}  prevalence {y.mean():.3f}")

# ----------------------------------------------------------------------
# 2. Custom class-weighted logistic regression (NumPy) with optional prox
# ----------------------------------------------------------------------
def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-35,35)))

def train_lr(Xa, ya, l2=1.0, lr=0.8, epochs=300, w0=None, b0=0.0, mu=0.0,
             wp=None, bp=0.0):
    n,p = Xa.shape
    w = np.zeros(p) if w0 is None else w0.copy()
    b = 0.0 if w0 is None else b0
    # class weights (balanced)
    pos = ya.sum(); neg = n-pos
    cw1 = n/(2*max(pos,1)); cw0 = n/(2*max(neg,1))
    sw = np.where(ya==1, cw1, cw0)
    swsum = sw.sum()
    for _ in range(epochs):
        z = Xa@w + b
        pr = sigmoid(z)
        g = (pr-ya)*sw
        gw = Xa.T@g/swsum + l2*w/n
        gb = g.sum()/swsum
        if mu>0 and wp is not None:        # FedProx proximal term
            gw += mu*(w-wp); gb += mu*(b-bp)
        w -= lr*gw; b -= lr*gb
    return w,b

def auc(w,b,Xa,ya):
    return roc_auc_score(ya, sigmoid(Xa@w+b))

# Centralized ceiling: scikit-learn L-BFGS reference solver (matches the parent
# benchmark's configuration: C=1.0, balanced class weight). FL methods below use
# SGD-style local training, which is standard in federated learning.
from sklearn.linear_model import LogisticRegression
_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000,
                         solver="lbfgs").fit(Xtr,ytr)
w_c = _lr.coef_.ravel().copy(); b_c = float(_lr.intercept_[0])
print(f"[ceiling] Centralized LR (sklearn LBFGS) test ROC-AUC = {auc(w_c,b_c,Xte,yte):.3f}")

# ----------------------------------------------------------------------
# 3. Client partition by institution archetype
# ----------------------------------------------------------------------
def archetype_score(row):
    """Higher -> more MFI-like (rural/agri/small); lower -> commercial."""
    return 0  # placeholder, replaced below

def partition(meta_df, Xpool, ypool, regime):
    """Return list of (Xk, yk, idxk) for 3 clients: MFI, Commercial, Cooperative."""
    n = len(ypool)
    loc = meta_df["location_type"].values
    sec = meta_df["sector"].values
    div = meta_df["division"].values
    # base archetype preference per record
    pref = np.zeros((n,3))  # cols: MFI, Commercial, Cooperative
    for i in range(n):
        mfi = 1.0*(loc[i]=="Rural") + 0.7*(sec[i]=="Agriculture") + 0.3*(sec[i]=="Retail")
        com = 1.0*(loc[i]=="Urban") + 0.6*(sec[i] in ("Manufacturing","Services"))
        coop = 1.0*(div[i] in ("Rangpur","Khulna","Barisal")) + 0.4*(loc[i]=="Rural")
        pref[i] = [mfi,com,coop]
    if regime=="high":
        assign = pref.argmax(1)
        # break ties / zeros randomly
        zero = pref.sum(1)==0
        assign[zero] = RNG.integers(0,3,zero.sum())
    else:
        bias = {"low":0.6,"medium":0.8}[regime]
        probs = np.full((n,3),(1-bias)/3)
        probs[np.arange(n),pref.argmax(1)] += bias
        probs /= probs.sum(1,keepdims=True)
        assign = np.array([RNG.choice(3,p=probs[i]) for i in range(n)])
    clients=[]
    for k in range(3):
        m = assign==k
        if m.sum()<10:  # guard
            continue
        clients.append((Xpool[m], ypool[m], np.where(m)[0]))
    return clients, assign

def partition_dirichlet(Xpool, ypool, alpha, K=3):
    """Standard label-skew Dirichlet partition for comparison with FL lit."""
    clients_idx=[[] for _ in range(K)]
    for c in [0,1]:
        ids = np.where(ypool==c)[0]; RNG.shuffle(ids)
        prop = RNG.dirichlet([alpha]*K)
        cuts = (np.cumsum(prop)*len(ids)).astype(int)[:-1]
        for k,part in enumerate(np.split(ids,cuts)):
            clients_idx[k]+=list(part)
    return [(Xpool[np.array(ix)],ypool[np.array(ix)],np.array(ix)) for ix in clients_idx if len(ix)>=10]

# ----------------------------------------------------------------------
# 4. Federated training
# ----------------------------------------------------------------------
def federated(clients, method="fedavg", mu=0.0, rounds=40, local_epochs=40):
    w = np.zeros(d); b=0.0
    ntot = sum(len(c[1]) for c in clients)
    for t in range(rounds):
        ws=[]; bs=[]; ns=[]
        for Xk,yk,_ in clients:
            wp = w.copy(); bp=b
            wk,bk = train_lr(Xk,yk,epochs=local_epochs,w0=w,b0=b,
                             mu=(mu if method=="fedprox" else 0.0),wp=wp,bp=bp)
            ws.append(wk); bs.append(bk); ns.append(len(yk))
        ns=np.array(ns,float); ns/=ns.sum()
        w = np.sum([nk*wk for nk,wk in zip(ns,ws)],axis=0)
        b = float(np.sum([nk*bk for nk,bk in zip(ns,bs)]))
    return w,b

def local_models(clients, epochs=1500):
    return [train_lr(Xk,yk,epochs=epochs) for Xk,yk,_ in clients]

# ----------------------------------------------------------------------
# 5. Analytic linear SHAP and stability metrics
# ----------------------------------------------------------------------
def lin_shap(w, Xa, baseline):
    """Exact SHAP for linear model: phi_j(x) = w_j*(x_j - baseline_j)."""
    return (Xa - baseline) * w   # (n,d)

def ranking(phi):  # mean abs over instances -> importance vector
    return np.abs(phi).mean(0)

def topk(vec,k): return set(np.argsort(-vec)[:k])

def jaccard(a,b):
    return len(a&b)/len(a|b) if (a|b) else 1.0

def cosine_rows(A,B):
    num=(A*B).sum(1); da=np.linalg.norm(A,axis=1); db=np.linalg.norm(B,axis=1)
    den=da*db; ok=den>1e-12
    out=np.zeros(len(A)); out[ok]=num[ok]/den[ok]
    return out

def stability_suite(w_glob,b_glob, clients, local_ms, m_top=5):
    """Compute S_cc, S_rank, J5, violation rate across clients."""
    rks=[]; cc_list=[]; viol=0; ntot=0
    for (Xk,yk,_),(wk,bk) in zip(clients,local_ms):
        mu_k = Xk.mean(0)                       # client interventional baseline
        phi_g = lin_shap(w_glob, Xk, mu_k)      # global model on client data
        phi_l = lin_shap(wk,     Xk, mu_k)      # local model on client data
        rks.append(ranking(phi_g))
        cc_list.append(cosine_rows(phi_g,phi_l).mean())
        # per-instance top-m consistency (global vs local)
        for i in range(len(yk)):
            if topk(np.abs(phi_g[i]),m_top)!=topk(np.abs(phi_l[i]),m_top):
                viol+=1
            ntot+=1
    # cross-client rank stability (global model, different client populations)
    sps=[]; js=[]
    for a,bb in itertools.combinations(range(len(rks)),2):
        sps.append(spearmanr(rks[a],rks[bb]).correlation)
        js.append(jaccard(topk(rks[a],m_top),topk(rks[bb],m_top)))
    return dict(S_cc=float(np.mean(cc_list)),
                S_rank=float(np.mean(sps)),
                J5=float(np.mean(js)),
                violation=float(viol/max(ntot,1)),
                rankings=np.array(rks))

def input_perturb_stability(w,b,Xa,sigma=0.1,reps=10,ninst=100):
    sub=Xa[:ninst]; base=Xa.mean(0)
    phi0=lin_shap(w,sub,base); sims=[]
    for _ in range(reps):
        noise=RNG.normal(0,sigma,size=(len(sub),len(NUM)))
        Xp=sub.copy(); Xp[:,:len(NUM)]=Xp[:,:len(NUM)]+noise
        phi=lin_shap(w,Xp,base)
        sims.append(cosine_rows(phi0,phi).mean())
    return float(np.mean(sims)),float(np.std(sims))

# ----------------------------------------------------------------------
# 6. Run across heterogeneity regimes
# ----------------------------------------------------------------------
rows=[]
regime_rank={}
per_feature_std={}
for regime in ["low","medium","high"]:
    clients,assign = partition(meta_tr,Xtr,ytr,regime)
    locs = local_models(clients)
    # centralized ceiling (same for all, but recompute stability vs these clients)
    cen = stability_suite(w_c,b_c,clients,locs)
    w_fa,b_fa = federated(clients,"fedavg")
    fa = stability_suite(w_fa,b_fa,clients,locs)
    w_fp,b_fp = federated(clients,"fedprox",mu=0.1)
    fp = stability_suite(w_fp,b_fp,clients,locs)
    # isolated local: stability of local vs local across clients = floor (use locs as both)
    iso = stability_suite_floor = None
    # accuracy on global test
    accs = dict(cen=auc(w_c,b_c,Xte,yte),
                fedavg=auc(w_fa,b_fa,Xte,yte),
                fedprox=auc(w_fp,b_fp,Xte,yte))
    # isolated: mean test AUC of local models
    accs["isolated"]=float(np.mean([auc(wk,bk,Xte,yte) for wk,bk in locs]))
    for tag,res,acc in [("Centralized",cen,accs["cen"]),
                        ("FedProx(mu=0.1)",fp,accs["fedprox"]),
                        ("FedAvg",fa,accs["fedavg"])]:
        rows.append(dict(regime=regime,setting=tag,test_auc=round(acc,4),
                         S_cc=round(res["S_cc"],4),S_rank=round(res["S_rank"],4),
                         J5=round(res["J5"],4),violation=round(res["violation"],4)))
    regime_rank[regime]=fa["rankings"]
    # per-feature cross-client std of attribution importance (FedAvg)
    per_feature_std[regime]=fa["rankings"].std(0)
    print(f"[{regime}] FedAvg AUC={accs['fedavg']:.3f} S_cc={fa['S_cc']:.3f} "
          f"S_rank={fa['S_rank']:.3f} J5={fa['J5']:.3f} viol={fa['violation']:.3f} | "
          f"FedProx S_rank={fp['S_rank']:.3f} viol={fp['violation']:.3f} | "
          f"Cen S_rank={cen['S_rank']:.3f}")

res_df=pd.DataFrame(rows)
res_df.to_csv(f"{OUT}/federated_stability.csv",index=False)
print("\n=== FEDERATED STABILITY TABLE ===")
print(res_df.to_string(index=False))

# Dirichlet comparison
dir_rows=[]
for a in [0.1,0.5,5.0]:
    cl=partition_dirichlet(Xtr,ytr,a)
    lc=local_models(cl)
    wfa,bfa=federated(cl,"fedavg")
    r=stability_suite(wfa,bfa,cl,lc)
    dir_rows.append(dict(alpha=a,n_clients=len(cl),test_auc=round(auc(wfa,bfa,Xte,yte),4),
                         S_cc=round(r["S_cc"],4),S_rank=round(r["S_rank"],4),
                         J5=round(r["J5"],4),violation=round(r["violation"],4)))
pd.DataFrame(dir_rows).to_csv(f"{OUT}/dirichlet_stability.csv",index=False)
print("\n=== DIRICHLET SKEW (FedAvg) ===")
print(pd.DataFrame(dir_rows).to_string(index=False))

# input perturbation stability (global FedAvg, high regime)
ip_m,ip_s=input_perturb_stability(w_fa,b_fa,Xte)
print(f"\nInput-perturbation stability (FedAvg global, sigma=0.1): {ip_m:.4f} +/- {ip_s:.4f}")

# ----------------------------------------------------------------------
# 7. Fairness audit: FedAvg global vs Centralized on global test
# ----------------------------------------------------------------------
def fairness(w,b,Xa,ya,meta_df,thr=0.3):
    pr=sigmoid(Xa@w+b); pred=(pr>=thr).astype(int)
    out=[]
    for attr in ["sector","division","location_type","owner_education"]:
        groups=meta_df[attr].values; dp=[];fnr=[];fpr=[];aucs=[]
        for g in pd.unique(groups):
            mask=groups==g
            if mask.sum()<5: continue
            yy=ya[mask]; pp=pred[mask]; sc=pr[mask]
            dp.append(pp.mean())
            pos=yy==1; neg=yy==0
            fnr.append(((pp==0)&pos).sum()/max(pos.sum(),1))
            fpr.append(((pp==1)&neg).sum()/max(neg.sum(),1))
            if len(np.unique(yy))>1: aucs.append(roc_auc_score(yy,sc))
        out.append(dict(attribute=attr,DP_gap=round(max(dp)-min(dp),4),
                        EO_gap=round(max(fnr)-min(fnr),4),
                        PE_gap=round(max(fpr)-min(fpr),4),
                        auc_min=round(min(aucs),4),auc_max=round(max(aucs),4)))
    return pd.DataFrame(out)

fair_fa=fairness(w_fa,b_fa,Xte,yte,meta_te); fair_fa["model"]="FedAvg"
fair_cen=fairness(w_c,b_c,Xte,yte,meta_te); fair_cen["model"]="Centralized"
fair=pd.concat([fair_cen,fair_fa])
fair.to_csv(f"{OUT}/federated_fairness.csv",index=False)
print("\n=== FAIRNESS (Centralized vs FedAvg, thr=0.3) ===")
print(fair.to_string(index=False))

# ----------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# (b) S_rank vs heterogeneity
plt.figure(figsize=(6,4))
order=["low","medium","high"]
for tag in ["Centralized","FedProx(mu=0.1)","FedAvg"]:
    ys=[res_df[(res_df.regime==r)&(res_df.setting==tag)].S_rank.values[0] for r in order]
    plt.plot(order,ys,marker="o",label=tag)
plt.ylabel("Cross-client rank stability $S_{rank}$"); plt.xlabel("Heterogeneity regime")
plt.title("Explanation rank stability vs client heterogeneity"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/figB_stability_vs_heterogeneity.png",dpi=150); plt.close()

# (c) per-feature cross-client attribution std (high regime, top 15)
std_high=per_feature_std["high"]; ordr=np.argsort(-std_high)[:15]
plt.figure(figsize=(7,5))
plt.barh([feat_names[i] for i in ordr][::-1], std_high[ordr][::-1])
plt.xlabel("Cross-client std of mean|SHAP| (FedAvg, high heterogeneity)")
plt.title("Most client-unstable features"); plt.tight_layout()
plt.savefig(f"{OUT}/figC_per_feature_instability.png",dpi=150); plt.close()

# (d) accuracy vs S_cc tradeoff (high regime)
plt.figure(figsize=(6,4))
hi=res_df[res_df.regime=="high"]
plt.scatter(hi.S_cc,hi.test_auc,s=80)
for _,r in hi.iterrows(): plt.annotate(r.setting,(r.S_cc,r.test_auc),fontsize=8,
                                       xytext=(5,5),textcoords="offset points")
plt.xlabel("Cross-client attribution stability $S_{cc}$"); plt.ylabel("Global test ROC-AUC")
plt.title("Accuracy vs explanation stability (high heterogeneity)"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/figD_accuracy_stability_tradeoff.png",dpi=150); plt.close()

# (a) per-client top-feature heatmap (FedAvg, high regime)
clients_h,_=partition(meta_tr,Xtr,ytr,"high")
locs_h=local_models(clients_h)
w_fah,b_fah=federated(clients_h,"fedavg")
rk=[]; labels=["MFI","Commercial","Cooperative"]
for (Xk,yk,_) in clients_h:
    rk.append(ranking(lin_shap(w_fah,Xk,Xk.mean(0))))
rk=np.array(rk)
topfeat=np.argsort(-rk.mean(0))[:12]
plt.figure(figsize=(8,5))
plt.imshow(rk[:,topfeat],aspect="auto",cmap="viridis")
plt.yticks(range(len(labels)),labels[:len(clients_h)])
plt.xticks(range(len(topfeat)),[feat_names[i] for i in topfeat],rotation=60,ha="right",fontsize=8)
plt.colorbar(label="mean|SHAP| (global model on client data)")
plt.title("Per-client feature attribution under FedAvg (high heterogeneity)")
plt.tight_layout(); plt.savefig(f"{OUT}/figA_client_attribution_heatmap.png",dpi=150); plt.close()

# save machine-readable summary
summary=dict(feature_dims=int(d),
             centralized_test_auc=round(float(auc(w_c,b_c,Xte,yte)),4),
             input_perturb_stability=[round(ip_m,4),round(ip_s,4)],
             stability_table=res_df.to_dict("records"),
             dirichlet=pd.DataFrame(dir_rows).to_dict("records"))
json.dump(summary,open(f"{OUT}/summary_federated.json","w"),indent=2)
print(f"\nAll outputs written to {OUT}/")
print("Figures: figA..figD ; CSVs: federated_stability, dirichlet_stability, federated_fairness")
