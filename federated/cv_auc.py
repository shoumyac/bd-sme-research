"""5-fold cross-validated ROC-AUC of the pooled model and each aggregator.
Preprocessing is refitted on each fold's training rows (no validation-row statistics)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import federated_xai_experiment_v2 as v2
Yf=v2.y; metaf=v2.meta; IDX=np.arange(len(Yf))
MODELS={"LR":v2.LRModel(),"MLP":v2.MLPModel()}
AGG=["centralized","fedavg","fedprox","scaffold","median","isolated"]
def cv_auc(model,regime,agg):
    skf=StratifiedKFold(5,shuffle=True,random_state=42);a=[]
    for tr,te in skf.split(IDX,Yf):
        Xtr,Xte=v2.fold_data(tr,te); ytr,yte=Yf[tr],Yf[te]; mt=metaf.iloc[tr].reset_index(drop=True)
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
