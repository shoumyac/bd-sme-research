import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import federated_xai_experiment_v2 as v2
Yf,metaf=v2.y,v2.meta; IDX=np.arange(len(Yf))
LR=v2.LRModel(); MLP=v2.MLPModel()
def folds():
    # preprocessing refitted on each fold's training rows only
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(IDX,Yf):
        A,B=v2.fold_data(tr,te); yield tr,te,A,B
# scaling
print("== scaling ==")
for mn,model in [("LR",LR),("MLP",MLP)]:
    for K in [3,6,10]:
        a=[]
        for tr,te,XA,XB in folds():
            mt=metaf.iloc[tr].reset_index(drop=True)
            cl=v2.partition(mt,XA,Yf[tr],"high",K=K); p=v2.federated(model,cl,"fedavg")
            a.append(roc_auc_score(Yf[te],model.prob(p,XB)))
        print(mn,K,round(np.mean(a),4))
# dp
print("== dp (LR) ==")
for sigma in [0.0,0.01,0.05,0.1,0.2]:
    a=[]
    for tr,te,XA,XB in folds():
        mt=metaf.iloc[tr].reset_index(drop=True); cl=v2.partition(mt,XA,Yf[tr],"high")
        p=LR.init()
        for t in range(20):
            ps=[v2.local_train(LR,p,Xk,yk,18,0.5) for Xk,yk,_ in cl]; w=np.array([len(c[1]) for c in cl],float);w/=w.sum()
            p=np.sum([wi*pi for wi,pi in zip(w,ps)],0)
            if sigma>0: p=p+np.random.default_rng(42).normal(0,sigma,p.shape)
        a.append(roc_auc_score(Yf[te],LR.prob(p,XB)))
    print(sigma,round(np.mean(a),4))
# dirichlet
print("== dirichlet ==")
for mn,model in [("LR",LR),("MLP",MLP)]:
    for alpha in [0.1,0.5,5.0]:
        a=[]
        for tr,te,XA,XB in folds():
            cl=v2.partition_dir(XA,Yf[tr],alpha); p=v2.federated(model,cl,"fedavg")
            a.append(roc_auc_score(Yf[te],model.prob(p,XB)))
        print(mn,alpha,round(np.mean(a),4))
# pfl
print("== pfl ==")
for mn,model in [("LR",LR),("MLP",MLP)]:
    ag=[];ap=[]
    for tr,te,XA,XB in folds():
        mt=metaf.iloc[tr].reset_index(drop=True); cl=v2.partition(mt,XA,Yf[tr],"high")
        pg=v2.federated(model,cl,"fedavg"); pers=[v2.local_train(model,pg.copy(),Xk,yk,12,0.3) for Xk,yk,_ in cl]
        ag.append(roc_auc_score(Yf[te],model.prob(pg,XB)))
        ap.append(np.mean([roc_auc_score(Yf[te],model.prob(pp,XB)) for pp in pers]))
    print(mn,"global",round(np.mean(ag),4),"pers",round(np.mean(ap),4))
