#!/usr/bin/env python3
"""
v3 extensions: (A) Vertical FL explanation fragmentation, (B) Personalized FL
explanation fidelity-vs-consistency tension. Reuses v2 machinery. random_state=42.
    python3 federated_xai_experiment_v3.py vfl
    python3 federated_xai_experiment_v3.py pfl
"""
import sys, os, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
import federated_xai_experiment_v2 as v2
OUT="results_federated_v3"; os.makedirs(OUT,exist_ok=True)
Xtr,ytr,Xte,yte=v2.Xtr,v2.ytr,v2.Xte,v2.yte
feat=v2.feat; d=v2.d; meta_tr=v2.meta_tr
LR=v2.LRModel(); MLP=v2.MLPModel()
def topk(v,k): return set(np.argsort(-v)[:k])

# ----- map one-hot columns to 3 vertical parties by semantics -----
def party_of(fname):
    A={"monthly_revenue_bdt","monthly_profit_bdt","loan_amount_bdt","preferred_tenure_months",
       "employees_count","business_age_years","days_payable_outstanding"}   # Financials
    if fname in A: return 0
    if fname.startswith(("prior_loan_count","cashflow_stability","repayment_confidence",
                         "existing_loans","late_payment_history","collateral_level")): return 1
    return 2   # demographics / geography / legal / sector / education
parties=np.array([party_of(f) for f in feat]); PNAMES=["Financials","Credit-behaviour","Demographic/Geo"]

def stage_vfl():
    rows=[]
    for mname,model in [("LR",LR),("MLP",MLP)]:
        p=v2.local_train(model,model.init(),Xtr,ytr,400,0.5)   # global VFL model (all features)
        base=Xtr.mean(0); A=model.attribution(p,Xte,base)      # global attributions on test
        # (1) reason fragmentation + single-party coverage over top-5
        frag=[]; cover=[]
        for i in range(len(Xte)):
            t5=topk(np.abs(A[i]),5); pset=parties[list(t5)]
            frag.append(len(set(pset)))
            cover.append(max((pset==pp).sum() for pp in range(3))/5.0)
        # (2) party-local rank fidelity: train model on each party's block, compare within-block importance
        fid=[]
        for pp in range(3):
            cols=np.where(parties==pp)[0]
            if mname=="LR":
                from sklearn.linear_model import LogisticRegression
                m=LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000).fit(Xtr[:,cols],ytr)
                loc_imp=np.abs(m.coef_.ravel())
            else:
                from sklearn.neural_network import MLPClassifier
                m=MLPClassifier(hidden_layer_sizes=(32,),max_iter=400,random_state=42).fit(Xtr[:,cols],ytr)
                # permutation importance proxy
                bp=m.predict_proba(Xtr[:,cols])[:,1]; base_auc=__import__("sklearn").metrics.roc_auc_score(ytr,bp)
                loc_imp=[]
                for j in range(len(cols)):
                    Xp=Xtr[:,cols].copy(); Xp[:,j]=np.random.permutation(Xp[:,j])
                    loc_imp.append(base_auc-__import__("sklearn").metrics.roc_auc_score(ytr,m.predict_proba(Xp)[:,1]))
                loc_imp=np.abs(np.array(loc_imp))
            glob_imp=np.abs(A[:,cols]).mean(0)
            if len(cols)>2: fid.append(spearmanr(loc_imp,glob_imp).correlation)
        rows.append(dict(model=mname,
            reasons_per_instance=round(np.mean(frag),3),
            single_party_coverage=round(np.mean(cover),3),
            party_local_vs_global_rho=round(np.nanmean(fid),3)))
        print(mname,"VFL: frag=%.2f cover=%.2f rho=%.2f"%(np.mean(frag),np.mean(cover),np.nanmean(fid)))
    pd.DataFrame(rows).to_csv(f"{OUT}/vfl_fragmentation.csv",index=False); print("vfl saved")

def stage_pfl():
    rows=[]
    for mname,model in [("LR",LR),("MLP",MLP)]:
        cl=v2.partition(meta_tr,Xtr,ytr,"high")
        locs=[v2.local_train(model,model.init(),Xk,yk,300,0.5) for Xk,yk,_ in cl]
        pglob=v2.federated(model,cl,"fedavg")
        pers=[v2.local_train(model,pglob.copy(),Xk,yk,12,0.3) for Xk,yk,_ in cl]   # personalize: fine-tune global
        # local fidelity: attribution(personalized vs local-only) vs (global vs local-only)
        def fidelity(pset):
            v=[]
            for (Xk,yk,_),pp,pl in zip(cl,pset,locs):
                mu=Xk.mean(0)
                v.append(v2.cos_rows(model.attribution(pp,Xk[:50],mu),model.attribution(pl,Xk[:50],mu)).mean())
            return np.mean(v)
        fid_glob=fidelity([pglob]*len(cl)); fid_pers=fidelity(pers)
        # cross-client consistency on shared eval (test set): pairwise top-5 disagreement across the client MODELS
        def crossviol(pset):
            base=Xte.mean(0); atts=[model.attribution(pp,Xte,base) for pp in pset]; vio=0;nt=0
            for a,b in itertools.combinations(range(len(pset)),2):
                for i in range(len(Xte)):
                    if topk(np.abs(atts[a][i]),5)!=topk(np.abs(atts[b][i]),5): vio+=1
                    nt+=1
            return vio/nt
        cv_glob=crossviol([pglob]*len(cl)); cv_pers=crossviol(pers)
        accg=np.mean([v2.aucp(model,pglob) for _ in cl]); accp=np.mean([v2.aucp(model,pp) for pp in pers])
        rows.append(dict(model=mname,
            local_fidelity_global=round(fid_glob,3), local_fidelity_personalized=round(fid_pers,3),
            crossclient_viol_global=round(cv_glob,3), crossclient_viol_personalized=round(cv_pers,3),
            auc_global=round(accg,3), auc_personalized=round(accp,3)))
        print(mname,"PFL: fid g=%.2f p=%.2f | crossviol g=%.2f p=%.2f"%(fid_glob,fid_pers,cv_glob,cv_pers))
    pd.DataFrame(rows).to_csv(f"{OUT}/pfl_tension.csv",index=False); print("pfl saved")

if __name__=="__main__":
    st=sys.argv[1] if len(sys.argv)>1 else "all"
    {"vfl":stage_vfl,"pfl":stage_pfl}.get(st,lambda:[stage_vfl(),stage_pfl()])()
