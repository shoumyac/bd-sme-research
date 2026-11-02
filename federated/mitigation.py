import numpy as np, pandas as pd, itertools
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import federated_xai_experiment_v2 as v2
OUT="results_federated_v2"
Xtr,ytr,metaf=v2.Xtr,v2.ytr,v2.meta_tr; feat=v2.feat
LR=v2.LRModel(); MLP=v2.MLPModel()
def topk(v,k): return set(np.argsort(-v)[:k])
def jac(a,b): return len(a&b)/len(a|b) if (a|b) else 1.0
def cos_rows(A,B):
    num=(A*B).sum(1);den=np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1);ok=den>1e-12
    o=np.zeros(len(A));o[ok]=num[ok]/den[ok];return o

def fedxs(model,pg,clients,locs,baseline=None,m=5,cap=60):
    """baseline=None -> per-client mean (status quo); else shared baseline vector."""
    rks=[];cc=[];vio=0;nt=0
    for (Xk,yk,_),pl in zip(clients,locs):
        Xs=Xk[:cap]; mu = Xk.mean(0) if baseline is None else baseline
        ag=model.attribution(pg,Xs,mu); al=model.attribution(pl,Xs,mu)
        rks.append(np.abs(model.attribution(pg,Xk,mu)).mean(0)); cc.append(cos_rows(ag,al).mean())
        for i in range(len(Xs)):
            if topk(np.abs(ag[i]),m)!=topk(np.abs(al[i]),m): vio+=1
            nt+=1
    sp=[];j5=[]
    for a,b in itertools.combinations(range(len(rks)),2):
        sp.append(spearmanr(rks[a],rks[b]).correlation); j5.append(jac(topk(rks[a],5),topk(rks[b],5)))
    return dict(S_cc=np.mean(cc),S_rank=np.mean(sp),J5=np.mean(j5),violation=vio/max(nt,1))

rows=[]
shared=Xtr.mean(0)
for mn,model in [("LR",LR),("MLP",MLP)]:
    for regime in ["high"]:
        cl=v2.partition(metaf,Xtr,ytr,regime); locs=[v2.local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
        pg=v2.federated(model,cl,"fedavg")
        a=fedxs(model,pg,cl,locs,baseline=None)      # per-client baseline
        b=fedxs(model,pg,cl,locs,baseline=shared)    # shared baseline mitigation
        rows.append(dict(model=mn,setting="per-client baseline",**{k:round(v,4) for k,v in a.items()}))
        rows.append(dict(model=mn,setting="shared baseline (mitigation)",**{k:round(v,4) for k,v in b.items()}))
        print(mn,"per-client viol=%.3f S_rank=%.3f | shared viol=%.3f S_rank=%.3f"%(a["violation"],a["S_rank"],b["violation"],b["S_rank"]))
# Dirichlet severe for both, with mitigation
for mn,model in [("LR",LR),("MLP",MLP)]:
    cl=v2.partition_dir(Xtr,ytr,0.1); locs=[v2.local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
    pg=v2.federated(model,cl,"fedavg")
    a=fedxs(model,pg,cl,locs,None); b=fedxs(model,pg,cl,locs,shared)
    rows.append(dict(model=mn,setting="Dirichlet a=0.1 per-client",**{k:round(v,4) for k,v in a.items()}))
    rows.append(dict(model=mn,setting="Dirichlet a=0.1 shared (mitig.)",**{k:round(v,4) for k,v in b.items()}))
    print(mn,"Dir per-client viol=%.3f | shared viol=%.3f"%(a["violation"],b["violation"]))
pd.DataFrame(rows).to_csv(f"{OUT}/mitigation.csv",index=False)

# Proposition check: per-feature Var_k(mu_k) vs per-feature attribution instability (LR, high)
cl=v2.partition(metaf,Xtr,ytr,"high")
mus=np.array([Xk.mean(0) for Xk,_,_ in cl]); base_var=mus.var(0)            # Var_k(mu_k)
pg=v2.federated(LR,cl,"fedavg")
attr_std=np.array([np.abs(LR.attribution(pg,Xk,Xk.mean(0))).mean(0) for Xk,_,_ in cl]).std(0)
rho=spearmanr(base_var,attr_std).correlation
pd.DataFrame({"feature":feat,"baseline_variance":np.round(base_var,4),"attr_instability":np.round(attr_std,4)}).to_csv(f"{OUT}/proposition_check.csv",index=False)
print("Proposition: Spearman(Var_k(mu_k), attribution instability) = %.3f"%rho)
