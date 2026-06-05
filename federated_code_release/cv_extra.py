import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import federated_xai_experiment_v2 as v2
X=np.vstack([v2.Xtr,v2.Xte]); Y=np.concatenate([v2.ytr,v2.yte])
META=pd.concat([v2.meta_tr, v2.meta.iloc[v2.te].reset_index(drop=True)],ignore_index=True) if hasattr(v2,'te') else None
# rebuild full meta aligned to X order (Xtr then Xte)
import pandas as pd
meta_full=pd.concat([v2.meta_tr, v2.meta.iloc[v2.te if hasattr(v2,'te') else []].reset_index(drop=True)],ignore_index=True)
# v2 exposes meta and the split; reconstruct:
meta_all=v2.meta
# We instead recompute folds over the original full set using v2.X,v2.y order
Xf=v2.X; Yf=v2.y; metaf=v2.meta
MODELS={"LR":v2.LRModel(),"MLP":v2.MLPModel()}
AGG=["centralized","fedavg","fedprox","scaffold","median","isolated"]
def cv_auc(model,regime,agg):
    skf=StratifiedKFold(5,shuffle=True,random_state=42);a=[]
    for tr,te in skf.split(Xf,Yf):
        Xtr,ytr,Xte,yte=Xf[tr],Yf[tr],Xf[te],Yf[te]; mt=metaf.iloc[tr].reset_index(drop=True)
        if agg=="centralized":
            p=v2.local_train(model,model.init(),Xtr,ytr,400,0.5)
        elif agg=="isolated":
            cl=v2.partition(mt,Xtr,ytr,regime); locs=[v2.local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
            a.append(np.mean([roc_auc_score(yte,model.prob(pl,Xte)) for pl in locs])); continue
        else:
            cl=v2.partition(mt,Xtr,ytr,regime); p=v2.federated(model,cl,agg)
        a.append(roc_auc_score(yte,model.prob(p,Xte)))
    return np.mean(a),np.std(a)
stage=sys.argv[1] if len(sys.argv)>1 else "high"
rows=[]
regimes=["low","medium","high"] if stage=="all" else [stage]
for mn,model in MODELS.items():
    for regime in regimes:
        for agg in AGG:
            m,s=cv_auc(model,regime,agg); rows.append(dict(model=mn,regime=regime,aggregator=agg,cv_auc=round(m,4),cv_std=round(s,4)))
            print(mn,regime,agg,"%.4f±%.4f"%(m,s))
pd.DataFrame(rows).to_csv(f"results_federated_v2/cv_auc_{stage}.csv",index=False)
