#!/usr/bin/env python3
"""
Federated Explainability Under Distribution Shift -- EXTENDED study (v2).
Run in stages to fit time limits:
    python3 federated_xai_experiment_v2.py zoo     # centralized 10-model comparison
    python3 federated_xai_experiment_v2.py grid    # 2 models x 3 regimes x 5 aggregators
    python3 federated_xai_experiment_v2.py extra    # client-scaling + DP ablation + Dirichlet
All stages write CSVs into results_federated_v2/. random_state = 42.
Attributions: exact linear SHAP for LR; Integrated Gradients for the MLP.
"""
import sys, os, time, itertools, warnings, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
np.random.seed(42); RNG=np.random.default_rng(42); t0=time.time()
OUT="results_federated_v2"; os.makedirs(OUT,exist_ok=True)

DATA="../data/bangladesh_sme_1200_records.csv"
if not os.path.exists(DATA): DATA="data/bangladesh_sme_1200_records.csv"
df=pd.read_csv(DATA); TARGET="target_missed_payment"
NUM=["business_age_years","employees_count","monthly_revenue_bdt","monthly_profit_bdt",
     "prior_loan_count","loan_amount_bdt","preferred_tenure_months","cashflow_stability",
     "repayment_confidence","days_payable_outstanding"]
CAT=["sector","legal_status","division","location_type","existing_loans",
     "late_payment_history","loan_purpose","collateral_level","owner_education"]
y=df[TARGET].astype(int).values
ct=ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
X=ct.fit_transform(df); X=X.toarray() if hasattr(X,"toarray") else X; X=X.astype(float)
feat=NUM+list(ct.named_transformers_["cat"].get_feature_names_out(CAT)); d=X.shape[1]
meta=df[["sector","division","location_type","owner_education"]].reset_index(drop=True)
idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=0.2,stratify=y,random_state=42)
Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
meta_tr=meta.iloc[tr].reset_index(drop=True)

def sig(z): return 1/(1+np.exp(-np.clip(z,-35,35)))
def relu(z): return np.maximum(0,z)
def classweights(yv):
    n=len(yv);pos=yv.sum();neg=n-pos
    return np.where(yv==1,n/(2*max(pos,1)),n/(2*max(neg,1)))

class LRModel:
    name="LR"
    def init(self): return np.zeros(d+1)
    def prob(self,p,X): return sig(X@p[:-1]+p[-1])
    def grad(self,p,X,yv,sw,l2=1.0):
        pr=self.prob(p,X);g=(pr-yv)*sw;s=sw.sum()
        return np.concatenate([X.T@g/s+l2*p[:-1]/len(yv),[g.sum()/s]])
    def attribution(self,p,X,base): return (X-base)*p[:-1]

H=32
class MLPModel:
    name="MLP"
    def init(self):
        rng=np.random.default_rng(0)
        return self.flat(rng.normal(0,np.sqrt(2/d),(d,H)),np.zeros(H),
                         rng.normal(0,np.sqrt(2/H),(H,)),0.0)
    def flat(self,W1,b1,W2,b2): return np.concatenate([W1.ravel(),b1,W2.ravel(),[b2]])
    def unflat(self,p):
        i=0;W1=p[i:i+d*H].reshape(d,H);i+=d*H;b1=p[i:i+H];i+=H;W2=p[i:i+H];i+=H;return W1,b1,W2,p[i]
    def prob(self,p,X):
        W1,b1,W2,b2=self.unflat(p);return sig(relu(X@W1+b1)@W2+b2)
    def grad(self,p,X,yv,sw,l2=1e-3):
        W1,b1,W2,b2=self.unflat(p);n=len(yv)
        z1=X@W1+b1;a1=relu(z1);pr=sig(a1@W2+b2)
        dl=((pr-yv)*sw)/sw.sum()
        gW2=a1.T@dl+l2*W2/n;gb2=dl.sum()
        dz1=np.outer(dl,W2)*(z1>0)
        return self.flat(X.T@dz1+l2*W1/n,dz1.sum(0),gW2,gb2)
    def input_grad(self,p,X):
        W1,b1,W2,b2=self.unflat(p);z1=X@W1+b1;return ((z1>0)*W2)@W1.T
    def attribution(self,p,X,base,steps=12):
        base=base.reshape(1,-1);diff=X-base;ig=np.zeros_like(X)
        for a in (np.arange(1,steps+1)-0.5)/steps: ig+=self.input_grad(p,base+a*diff)
        return diff*ig/steps

def local_train(model,p0,X,yv,epochs,lr,mu=0.0,prox=None,corr=None):
    p=p0.copy();sw=classweights(yv)
    for _ in range(epochs):
        g=model.grad(p,X,yv,sw)
        if mu>0 and prox is not None: g=g+mu*(p-prox)
        if corr is not None: g=g+corr
        p=p-lr*g
    return p

def federated(model,clients,method="fedavg",mu=0.1,rounds=20,epochs=18,lr=0.5):
    p=model.init();c_glob=np.zeros_like(p);c_loc=[np.zeros_like(p) for _ in clients]
    for t in range(rounds):
        ps=[];ns=[];newc=[]
        for i,(Xk,yk,_) in enumerate(clients):
            if method=="scaffold":
                pk=local_train(model,p,Xk,yk,epochs,lr,corr=(c_glob-c_loc[i]))
                newc.append(c_loc[i]-c_glob+(p-pk)/(epochs*lr))
            elif method=="fedprox":
                pk=local_train(model,p,Xk,yk,epochs,lr,mu=mu,prox=p)
            else:
                pk=local_train(model,p,Xk,yk,epochs,lr)
            ps.append(pk);ns.append(len(yk))
        w=np.array(ns,float);w/=w.sum()
        p=np.median(np.array(ps),0) if method=="median" else np.sum([wi*pi for wi,pi in zip(w,ps)],0)
        if method=="scaffold":
            c_glob=c_glob+np.mean([nc-cl for nc,cl in zip(newc,c_loc)],0);c_loc=newc
    return p

def partition(meta_df,Xp,yp,regime,K=3):
    n=len(yp);loc=meta_df["location_type"].values;sec=meta_df["sector"].values;div=meta_df["division"].values
    pref=np.zeros((n,3))
    for i in range(n):
        pref[i,0]=1.0*(loc[i]=="Rural")+0.7*(sec[i]=="Agriculture")+0.3*(sec[i]=="Retail")
        pref[i,1]=1.0*(loc[i]=="Urban")+0.6*(sec[i] in ("Manufacturing","Services"))
        pref[i,2]=1.0*(div[i] in ("Rangpur","Khulna","Barisal"))+0.4*(loc[i]=="Rural")
    if regime=="high":
        asg=pref.argmax(1);z=pref.sum(1)==0;asg[z]=RNG.integers(0,3,z.sum())
    else:
        bias={"low":0.6,"medium":0.8}[regime];pr=np.full((n,3),(1-bias)/3)
        pr[np.arange(n),pref.argmax(1)]+=bias;pr/=pr.sum(1,keepdims=True)
        asg=np.array([RNG.choice(3,p=pr[i]) for i in range(n)])
    if K>3:
        sub=RNG.integers(0,K//3+1,n);asg=asg*(K//3+1)+np.minimum(sub,K//3)
    return [(Xp[asg==k],yp[asg==k],None) for k in np.unique(asg) if (asg==k).sum()>=15]

def partition_dir(Xp,yp,alpha,K=3):
    ci=[[] for _ in range(K)]
    for c in [0,1]:
        ids=np.where(yp==c)[0];RNG.shuffle(ids);pr=RNG.dirichlet([alpha]*K)
        cuts=(np.cumsum(pr)*len(ids)).astype(int)[:-1]
        for k,part in enumerate(np.split(ids,cuts)): ci[k]+=list(part)
    return [(Xp[np.array(ix)],yp[np.array(ix)],None) for ix in ci if len(ix)>=15]

def rank(a): return np.abs(a).mean(0)
def topk(v,k): return set(np.argsort(-v)[:k])
def jac(a,b): return len(a&b)/len(a|b) if (a|b) else 1.0
def cos_rows(A,B):
    num=(A*B).sum(1);den=np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)
    ok=den>1e-12;o=np.zeros(len(A));o[ok]=num[ok]/den[ok];return o
def aucp(model,p): return roc_auc_score(yte,model.prob(p,Xte))

def fedxs(model,pg,clients,locs,m=5,cap=50):
    rks=[];cc=[];vio=0;nt=0
    for (Xk,yk,_),pl in zip(clients,locs):
        Xs=Xk[:cap];mu=Xk.mean(0)
        ag=model.attribution(pg,Xs,mu);al=model.attribution(pl,Xs,mu)
        rks.append(rank(model.attribution(pg,Xk,mu)));cc.append(cos_rows(ag,al).mean())
        for i in range(len(Xs)):
            if topk(np.abs(ag[i]),m)!=topk(np.abs(al[i]),m): vio+=1
            nt+=1
    sp=[];kt=[];j3=[];j5=[];j10=[]
    for a,b in itertools.combinations(range(len(rks)),2):
        sp.append(spearmanr(rks[a],rks[b]).correlation);kt.append(kendalltau(rks[a],rks[b]).correlation)
        j3.append(jac(topk(rks[a],3),topk(rks[b],3)));j5.append(jac(topk(rks[a],5),topk(rks[b],5)))
        j10.append(jac(topk(rks[a],10),topk(rks[b],10)))
    return dict(S_cc=np.mean(cc),S_rank=np.mean(sp),Kendall=np.mean(kt),
                J3=np.mean(j3),J5=np.mean(j5),J10=np.mean(j10),violation=vio/max(nt,1),
                rankings=np.array(rks))

MODELS={"LR":LRModel(),"MLP":MLPModel()}
AGG=["centralized","fedavg","fedprox","scaffold","median"]
def cen_params(model): return local_train(model,model.init(),Xtr,ytr,400,0.5)

def stage_zoo():
    from sklearn.linear_model import LogisticRegression,SGDClassifier
    from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier,
        GradientBoostingClassifier,AdaBoostClassifier,HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    zoo={"Logistic Regression":lambda:LogisticRegression(C=1.0,class_weight="balanced",max_iter=10000),
     "Linear SVM (SGD)":lambda:SGDClassifier(loss="log_loss",alpha=1e-3,class_weight="balanced",max_iter=1500,random_state=42),
     "MLP (1x32)":lambda:MLPClassifier(hidden_layer_sizes=(32,),max_iter=400,random_state=42),
     "Random Forest":lambda:RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1),
     "Extra Trees":lambda:ExtraTreesClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1),
     "Gradient Boosting":lambda:GradientBoostingClassifier(random_state=42),
     "HistGradientBoosting":lambda:HistGradientBoostingClassifier(random_state=42),
     "AdaBoost":lambda:AdaBoostClassifier(n_estimators=150,random_state=42),
     "k-NN (k=15)":lambda:KNeighborsClassifier(n_neighbors=15),
     "Gaussian NB":lambda:GaussianNB()}
    skf=StratifiedKFold(5,shuffle=True,random_state=42);rows=[]
    for name,fn in zoo.items():
        au=[];pr=[];br=[]
        for a,b in skf.split(Xtr,ytr):
            m=fn();m.fit(Xtr[a],ytr[a]);p=m.predict_proba(Xtr[b])[:,1]
            au.append(roc_auc_score(ytr[b],p));pr.append(average_precision_score(ytr[b],p));br.append(brier_score_loss(ytr[b],p))
        rows.append(dict(Model=name,ROC_AUC=round(np.mean(au),4),ROC_AUC_std=round(np.std(au),4),
                         PR_AUC=round(np.mean(pr),4),Brier=round(np.mean(br),4)))
        print(f"[{time.time()-t0:.0f}s] {name}: AUC={np.mean(au):.3f}")
    pd.DataFrame(rows).sort_values("ROC_AUC",ascending=False).to_csv(f"{OUT}/model_zoo.csv",index=False)
    print("zoo saved")

def stage_grid():
    rows=[];perfeat={}
    for mn,model in MODELS.items():
        pcen=cen_params(model)
        for regime in ["low","medium","high"]:
            cl=partition(meta_tr,Xtr,ytr,regime)
            locs=[local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
            for agg in AGG:
                p=pcen if agg=="centralized" else federated(model,cl,agg)
                r=fedxs(model,p,cl,locs)
                rows.append(dict(model=mn,regime=regime,aggregator=agg,test_auc=round(aucp(model,p),4),
                    S_cc=round(r["S_cc"],4),S_rank=round(r["S_rank"],4),Kendall=round(r["Kendall"],4),
                    J3=round(r["J3"],4),J5=round(r["J5"],4),J10=round(r["J10"],4),violation=round(r["violation"],4)))
            rows.append(dict(model=mn,regime=regime,aggregator="isolated",
                test_auc=round(np.mean([aucp(model,pl) for pl in locs]),4),
                S_cc=np.nan,S_rank=np.nan,Kendall=np.nan,J3=np.nan,J5=np.nan,J10=np.nan,violation=np.nan))
            if regime=="high":
                pf=federated(model,cl,"fedavg")
                perfeat[mn]=np.array([rank(model.attribution(pf,Xk,Xk.mean(0))) for Xk,yk,_ in cl]).std(0)
            print(f"[{time.time()-t0:.0f}s] {mn} {regime} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/fed_stability_v2.csv",index=False)
    pf=[]
    for mn,std in perfeat.items():
        for i in np.argsort(-std)[:12]: pf.append(dict(model=mn,feature=feat[i],cross_client_std=round(float(std[i]),4)))
    pd.DataFrame(pf).to_csv(f"{OUT}/per_feature_instability_v2.csv",index=False)
    print("grid saved")

def stage_extra():
    sc=[]
    for mn,model in MODELS.items():
        for K in [3,5,10]:
            cl=partition(meta_tr,Xtr,ytr,"high",K=K)
            locs=[local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
            pf=federated(model,cl,"fedavg");r=fedxs(model,pf,cl,locs)
            sc.append(dict(model=mn,K=len(cl),test_auc=round(aucp(model,pf),4),
                S_rank=round(r["S_rank"],4),J5=round(r["J5"],4),violation=round(r["violation"],4)))
        print(f"[{time.time()-t0:.0f}s] scaling {mn} done")
    pd.DataFrame(sc).to_csv(f"{OUT}/client_scaling.csv",index=False)
    model=MODELS["LR"];cl=partition(meta_tr,Xtr,ytr,"high")
    locs=[local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl];dp=[]
    for sigma in [0.0,0.01,0.05,0.1,0.2]:
        p=model.init()
        for t in range(20):
            ps=[local_train(model,p,Xk,yk,18,0.5) for Xk,yk,_ in cl]
            w=np.array([len(c[1]) for c in cl],float);w/=w.sum()
            p=np.sum([wi*pi for wi,pi in zip(w,ps)],0)
            if sigma>0: p=p+RNG.normal(0,sigma,p.shape)
        r=fedxs(model,p,cl,locs)
        dp.append(dict(dp_sigma=sigma,test_auc=round(aucp(model,p),4),S_rank=round(r["S_rank"],4),
            J5=round(r["J5"],4),violation=round(r["violation"],4)))
    pd.DataFrame(dp).to_csv(f"{OUT}/dp_ablation.csv",index=False)
    di=[]
    for mn,model in MODELS.items():
        for alpha in [0.1,0.5,5.0]:
            cl=partition_dir(Xtr,ytr,alpha);locs=[local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
            pf=federated(model,cl,"fedavg");r=fedxs(model,pf,cl,locs)
            di.append(dict(model=mn,alpha=alpha,test_auc=round(aucp(model,pf),4),S_cc=round(r["S_cc"],4),
                S_rank=round(r["S_rank"],4),J5=round(r["J5"],4),violation=round(r["violation"],4)))
    pd.DataFrame(di).to_csv(f"{OUT}/dirichlet_v2.csv",index=False)
    print("extra saved")

if __name__=="__main__":
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    {"zoo":stage_zoo,"grid":stage_grid,"extra":stage_extra}.get(stage,lambda:[stage_zoo(),stage_grid(),stage_extra()])()
    print(f"[{time.time()-t0:.0f}s] stage '{stage}' complete")
