from pathlib import Path
import argparse, json, math, warnings, platform
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, accuracy_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 20260907
RNG = np.random.default_rng(SEED)
parser = argparse.ArgumentParser(description="Reproduce all analyses for the MetS hepatic-biomarker manuscript.")
parser.add_argument("--input", required=True, type=Path, help="Path to the de-identified source .xlsx file")
parser.add_argument("--sheet", default="Data", help="Worksheet containing participant-level data")
parser.add_argument("--output", default=Path("results"), type=Path, help="Output directory")
parser.add_argument("--seed", default=20260907, type=int, help="Random seed")
args = parser.parse_args()
SEED = args.seed
RNG = np.random.default_rng(SEED)
INPUT = args.input
OUTDIR = args.output
OUTDIR.mkdir(exist_ok=True, parents=True)

raw = pd.read_excel(INPUT, sheet_name=args.sheet)
raw.columns = ["ID","Sex","Age","Height_cm","Weight_kg","BMI","WC_cm","SBP_mmHg","DBP_mmHg",
               "FBG_mg_dL","TG_mg_dL","HDL_mg_dL","Stored_MetS","ALT_U_L","AST_U_L","GGT_U_L",
               "Stored_FLI","Stored_HSI"]
raw["Sex"] = raw["Sex"].astype(str).str.strip().str.upper()
raw["Female"] = (raw["Sex"] == "F").astype(int)

# Formula-based recalculation. No diabetes variable is available, and diagnosed
# cardiometabolic disease was an exclusion; therefore the diabetes increment in
# HSI is set to 0 and explicitly documented.
L = (0.953*np.log(raw["TG_mg_dL"]) + 0.139*raw["BMI"] + 0.718*np.log(raw["GGT_U_L"])
     + 0.053*raw["WC_cm"] - 15.745)
raw["FLI_recalc"] = 100*expit(L)
raw["HSI_recalc"] = 8*(raw["ALT_U_L"]/raw["AST_U_L"]) + raw["BMI"] + 2*raw["Female"]

# Harmonised MetS: any 3 of 5; central obesity is not mandatory.
raw["C_WC"] = (((raw.Sex=="M") & (raw.WC_cm>=90)) | ((raw.Sex=="F") & (raw.WC_cm>=80))).astype(int)
raw["C_TG"] = (raw.TG_mg_dL>=150).astype(int)
raw["C_HDL"] = (((raw.Sex=="M") & (raw.HDL_mg_dL<40)) | ((raw.Sex=="F") & (raw.HDL_mg_dL<50))).astype(int)
raw["C_BP"] = ((raw.SBP_mmHg>=130) | (raw.DBP_mmHg>=85)).astype(int)
raw["C_FBG"] = (raw.FBG_mg_dL>=100).astype(int)
raw["MetS_count"] = raw[["C_WC","C_TG","C_HDL","C_BP","C_FBG"]].sum(axis=1)
raw["MetS_recalc"] = (raw.MetS_count>=3).astype(int)
# Leakage-controlled analytic sensitivity outcome for FLI: >=2 of BP, FBG, HDL.
raw["MetS_star_count"] = raw[["C_BP","C_FBG","C_HDL"]].sum(axis=1)
raw["MetS_star"] = (raw.MetS_star_count>=2).astype(int)
raw["Stored_MetS_binary"] = raw.Stored_MetS.astype(str).str.lower().str.replace("-", "", regex=False).eq("mets").astype(int)
raw["Eligible_18_25"] = raw.Age.between(18,25).astype(int)
raw["ID_duplicate_flag"] = raw.ID.duplicated(False).astype(int)
raw["Stored_status_mismatch"] = (raw.Stored_MetS_binary != raw.MetS_recalc).astype(int)
raw["FLI_formula_mismatch_gt_0_01"] = ((raw.Stored_FLI-raw.FLI_recalc).abs()>0.01).astype(int)
raw["HSI_formula_mismatch_gt_0_01"] = ((raw.Stored_HSI-raw.HSI_recalc).abs()>0.01).astype(int)

# Primary analytic sample follows stated age eligibility and complete cases.
needed = ["Sex","Age","BMI","WC_cm","SBP_mmHg","DBP_mmHg","FBG_mg_dL","TG_mg_dL","HDL_mg_dL",
          "ALT_U_L","AST_U_L","GGT_U_L","FLI_recalc","HSI_recalc","MetS_recalc","MetS_star"]
d = raw.loc[(raw.Eligible_18_25==1) & raw[needed].notna().all(axis=1)].copy().reset_index(drop=True)

def auc_ci(y, score, B=1000, seed=SEED):
    y=np.asarray(y); score=np.asarray(score); auc=roc_auc_score(y,score); rng=np.random.default_rng(seed); vals=[]
    idx0=np.where(y==0)[0]; idx1=np.where(y==1)[0]
    for _ in range(B):
        ix=np.r_[rng.choice(idx0,len(idx0),replace=True),rng.choice(idx1,len(idx1),replace=True)]
        vals.append(roc_auc_score(y[ix],score[ix]))
    return auc, *np.quantile(vals,[.025,.975])

def cutoff_metrics(y, score, B=1000, seed=SEED):
    y=np.asarray(y); score=np.asarray(score)
    def once(yy,ss):
        fpr,tpr,thr=roc_curve(yy,ss); j=tpr-fpr; k=int(np.nanargmax(j)); cut=thr[k]
        pred=(ss>=cut).astype(int); tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
        return [cut,tp/(tp+fn),tn/(tn+fp),tp/(tp+fp) if tp+fp else np.nan,tn/(tn+fn) if tn+fn else np.nan,(tp+tn)/len(yy),j[k],tn,fp,fn,tp]
    main=once(y,score); rng=np.random.default_rng(seed); boot=[]; i0=np.where(y==0)[0];i1=np.where(y==1)[0]
    for _ in range(B):
        ix=np.r_[rng.choice(i0,len(i0),True),rng.choice(i1,len(i1),True)]
        boot.append(once(y[ix],score[ix])[:7])
    b=np.asarray(boot,float); cis=[]
    for j in range(7): cis.extend(np.nanquantile(b[:,j],[.025,.975]))
    return main,cis

def calibration_stats(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); lp=np.log(p/(1-p))
    # Calibration intercept with offset lp: solve sum(y-expit(a+lp))=0.
    res=minimize(lambda a: -np.sum(y*np.log(expit(a[0]+lp))+(1-y)*np.log(1-expit(a[0]+lp))),[0.0])
    intercept=float(res.x[0])
    # slope + free intercept.
    def nll(b):
        q=np.clip(expit(b[0]+b[1]*lp),1e-12,1-1e-12)
        return -np.sum(y*np.log(q)+(1-y)*np.log(1-q))
    rr=minimize(nll,[0,1],method="BFGS")
    return intercept,float(rr.x[1]),brier_score_loss(y,p)

def mle_logit(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float); X=np.c_[np.ones(len(X)),X]
    def nll(b):
        z=X@b
        return np.sum(np.logaddexp(0,z)-y*z)
    def grad(b): return X.T@(expit(X@b)-y)
    res=minimize(nll,np.zeros(X.shape[1]),jac=grad,method="BFGS")
    b=res.x; w=expit(X@b)*(1-expit(X@b)); H=X.T@(X*w[:,None]); cov=np.linalg.pinv(H); se=np.sqrt(np.diag(cov));
    z=b/se; p=2*norm.sf(abs(z)); return b,se,p

def cv_predictions(X,y,kind="LR"):
    X=np.asarray(X,float); y=np.asarray(y,int); outer=StratifiedKFold(5,shuffle=True,random_state=SEED); pred=np.zeros(len(y)); rows=[]; importances=[]
    if kind=="LR":
        pipe=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=5000,class_weight=None,solver="liblinear",random_state=SEED))])
        grid={"model__C":[0.01,0.1,1,10]}
    else:
        pipe=Pipeline([("model",RandomForestClassifier(class_weight=None,random_state=SEED,n_jobs=-1))])
        grid={"model__n_estimators":[150],"model__max_depth":[3,5],"model__min_samples_leaf":[5],"model__max_features":["sqrt"]}
    for fold,(tr,te) in enumerate(outer.split(X,y),1):
        inner=StratifiedKFold(4,shuffle=True,random_state=SEED+fold)
        gs=GridSearchCV(pipe,grid,scoring="roc_auc",cv=inner,n_jobs=-1,refit=True)
        gs.fit(X[tr],y[tr]); pred[te]=gs.predict_proba(X[te])[:,1]
        rows.append({"fold":fold,"best_params":json.dumps(gs.best_params_,sort_keys=True),"inner_auc":gs.best_score_})
        pi=permutation_importance(gs.best_estimator_,X[te],y[te],scoring="roc_auc",n_repeats=10,random_state=SEED+fold)
        importances.append(pi.importances_mean)
    return pred,rows,np.asarray(importances)

# Audit tables.
audit = pd.DataFrame([
    ["Rows in source file",len(raw),"Source workbook"],
    ["Complete rows",int(raw[needed].notna().all(axis=1).sum()),"No missing values in required fields"],
    ["Age outside 18-25",int((raw.Eligible_18_25==0).sum()),"Excluded from primary analysis"],
    ["Primary analytic N",len(d),"Age 18-25 and complete"],
    ["Stored MetS cases",int(raw.Stored_MetS_binary.sum()),"Stored label"],
    ["Recalculated MetS cases (all 403)",int(raw.MetS_recalc.sum()),"Any 3 of 5 harmonised criteria"],
    ["Recalculated MetS cases (primary)",int(d.MetS_recalc.sum()),"Any 3 of 5 harmonised criteria"],
    ["MetS* cases (primary)",int(d.MetS_star.sum()),">=2 of BP, FBG, HDL"],
    ["Stored status disagreements",int(raw.Stored_status_mismatch.sum()),"Stored vs recalculated MetS"],
    ["Duplicate ID rows",int(raw.ID_duplicate_flag.sum()),"Retained; numbering requires source check"],
    ["Unique duplicated ID values",int(raw.loc[raw.ID_duplicate_flag==1,"ID"].nunique()),"IDs 368, 369, 374"],
    ["FLI mismatches >0.01",int(raw.FLI_formula_mismatch_gt_0_01.sum()),"Stored vs published formula"],
    ["HSI mismatches >0.01",int(raw.HSI_formula_mismatch_gt_0_01.sum()),"Stored vs published formula; diabetes increment unavailable"],
    ["TG mean +/- SD",f"{raw.TG_mg_dL.mean():.2f} +/- {raw.TG_mg_dL.std():.2f}","Narrow dispersion confirmed in source data; verify laboratory/source records"],
],columns=["Check","Result","Interpretation"])

# Descriptive group comparisons with SMD.
vars_=["Age","BMI","WC_cm","SBP_mmHg","DBP_mmHg","FBG_mg_dL","TG_mg_dL","HDL_mg_dL","ALT_U_L","AST_U_L","GGT_U_L","FLI_recalc","HSI_recalc"]
desc=[]
for outcome in ["MetS_recalc","MetS_star"]:
    for v in vars_:
        a=d.loc[d[outcome]==1,v];b=d.loc[d[outcome]==0,v]
        sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        desc.append([outcome,v,len(a),a.mean(),a.std(),len(b),b.mean(),b.std(),(a.mean()-b.mean())/sp])
desc=pd.DataFrame(desc,columns=["Outcome","Variable","Cases_n","Cases_mean","Cases_SD","Noncases_n","Noncases_mean","Noncases_SD","SMD"])

# Sex-specific descriptive comparisons with Welch tests, SMDs, and
# Benjamini-Hochberg false-discovery-rate adjustment.
def bh_adjust(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]; n=len(p)
    adj=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.minimum(adj,1); return out
sexdesc=[]
for sex in ["F","M"]:
  dd=d[d.Sex==sex]
  tmp=[]
  for v in vars_:
    a=dd.loc[dd.MetS_recalc==1,v]; b=dd.loc[dd.MetS_recalc==0,v]
    from scipy.stats import ttest_ind
    tt=ttest_ind(a,b,equal_var=False,nan_policy="omit")
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    tmp.append([sex,v,len(a),a.mean(),a.std(),len(b),b.mean(),b.std(),tt.statistic,tt.pvalue,(a.mean()-b.mean())/sp])
  q=bh_adjust([x[9] for x in tmp])
  for x,qq in zip(tmp,q): sexdesc.append(x+[qq])
sexdesc=pd.DataFrame(sexdesc,columns=["Sex","Variable","MetS_n","MetS_mean","MetS_SD","NonMetS_n","NonMetS_mean","NonMetS_SD","Welch_t","P_value","SMD","FDR_q"])

# Pearson correlation matrix used in Figure 1.
corrvars=["ALT_U_L","AST_U_L","GGT_U_L","FLI_recalc","HSI_recalc","BMI","WC_cm","SBP_mmHg","DBP_mmHg","FBG_mg_dL","TG_mg_dL","HDL_mg_dL"]
corr=d[corrvars].corr(method="pearson")

# Single-marker ROC and exploratory thresholds overall and by sex.
markers=["ALT_U_L","AST_U_L","GGT_U_L","FLI_recalc","HSI_recalc"]
roc_rows=[]
for outcome in ["MetS_recalc","MetS_star"]:
  for subgroup,dd in [("Overall",d),("Female",d[d.Sex=="F"]),("Male",d[d.Sex=="M"])]:
    for mi,m in enumerate(markers):
      y=dd[outcome].to_numpy();s=dd[m].to_numpy(); auc,lo,hi=auc_ci(y,s,1000,SEED+mi)
      main,cis=cutoff_metrics(y,s,1000,SEED+mi+100)
      roc_rows.append([outcome,subgroup,m,len(dd),int(y.sum()),auc,lo,hi,*main[:7],*cis])
cols=["Outcome","Subgroup","Marker","N","Cases","AUC","AUC_L95","AUC_U95","Cutoff","Sensitivity","Specificity","PPV","NPV","Accuracy","Youden"]
for nm in ["Cutoff","Sensitivity","Specificity","PPV","NPV","Accuracy","Youden"]: cols += [nm+"_L95",nm+"_U95"]
rocres=pd.DataFrame(roc_rows,columns=cols)

# Formal age-adjusted marker-by-sex interaction tests. FLI is tested against MetS*;
# other markers against conventional MetS. Marker is z-standardised.
inter=[]
for m in markers:
    outcome="MetS_star" if m=="FLI_recalc" else "MetS_recalc"
    z=(d[m]-d[m].mean())/d[m].std(ddof=0); agez=(d.Age-d.Age.mean())/d.Age.std(ddof=0); sx=d.Female
    X=np.c_[z,sx,z*sx,agez]; b,se,p=mle_logit(X,d[outcome])
    inter.append([m,outcome,b[1],np.exp(b[1]),np.exp(b[1]-1.96*se[1]),np.exp(b[1]+1.96*se[1]),p[1],b[3],np.exp(b[3]),np.exp(b[3]-1.96*se[3]),np.exp(b[3]+1.96*se[3]),p[3]])
inter=pd.DataFrame(inter,columns=["Marker","Outcome","Marker_beta","Marker_OR_per_SD","Marker_OR_L95","Marker_OR_U95","Marker_p","Interaction_beta","Interaction_OR","Interaction_OR_L95","Interaction_OR_U95","Interaction_p"])

# Main adjusted association table. FLI is evaluated against MetS*; HSI is
# adjusted for age only because sex is already embedded in the HSI formula.
assoc=[]
for m in markers:
    outcome="MetS_star" if m=="FLI_recalc" else "MetS_recalc"
    z=(d[m]-d[m].mean())/d[m].std(ddof=0)
    agez=(d.Age-d.Age.mean())/d.Age.std(ddof=0)
    if m=="HSI_recalc":
        X=np.c_[z,agez]; adjustment="Age"
    else:
        X=np.c_[z,agez,d.Female]; adjustment="Age and sex"
    b,se,p=mle_logit(X,d[outcome])
    assoc.append([m,outcome,adjustment,np.exp(b[1]),np.exp(b[1]-1.96*se[1]),np.exp(b[1]+1.96*se[1]),p[1]])
associations=pd.DataFrame(assoc,columns=["Marker","Outcome","Adjustment","OR_per_SD","OR_L95","OR_U95","P_value"])

# Nested CV, all against leakage-controlled MetS* so there is no direct FLI-WC/TG overlap.
families={
 "Age+Sex base":["Age","Female"],
 "Enzymes only":["ALT_U_L","AST_U_L","GGT_U_L"],
 "FLI only":["FLI_recalc"],
 "HSI only":["HSI_recalc"],
 "Age+Sex+enzymes":["Age","Female","ALT_U_L","AST_U_L","GGT_U_L"],
 "Age+Sex+FLI":["Age","Female","FLI_recalc"],
 "Age+HSI (sex embedded)":["Age","HSI_recalc"],
}
model_rows=[]; fold_rows=[]; pred_store={}; perm_rows=[]
y=d.MetS_star.to_numpy()
for fi,(fam,features) in enumerate(families.items()):
  for kind in ["LR","RF"]:
    pred,folds,pims=cv_predictions(d[features].to_numpy(),y,kind)
    pred_store[(fam,kind)]=pred
    auc,lo,hi=auc_ci(y,pred,1000,SEED+fi+500); calint,calslope,brier=calibration_stats(y,pred)
    main,_=cutoff_metrics(y,pred,400,SEED+fi+800)
    model_rows.append([fam,kind,", ".join(features),len(y),int(y.sum()),auc,lo,hi,brier,calint,calslope,*main[:7]])
    for rr in folds: fold_rows.append([fam,kind,rr["fold"],rr["inner_auc"],rr["best_params"]])
    for j,f in enumerate(features): perm_rows.append([fam,kind,f,float(pims[:,j].mean()),float(pims[:,j].std(ddof=1))])
modelcols=["Model_family","Algorithm","Features","N","Cases","AUC","AUC_L95","AUC_U95","Brier","Calibration_intercept","Calibration_slope","Cutoff","Sensitivity","Specificity","PPV","NPV","Accuracy","Youden"]
modelres=pd.DataFrame(model_rows,columns=modelcols); foldres=pd.DataFrame(fold_rows,columns=["Model_family","Algorithm","Outer_fold","Best_inner_AUC","Best_parameters"]); permres=pd.DataFrame(perm_rows,columns=["Model_family","Algorithm","Feature","Permutation_AUC_drop_mean","Permutation_AUC_drop_SD"])

# Final out-of-fold confusion matrices at each model's Youden cutoff.
confrows=[]
for (fam,kind),pred in pred_store.items():
    mr=modelres[(modelres.Model_family==fam)&(modelres.Algorithm==kind)].iloc[0]
    yp=(pred>=mr.Cutoff).astype(int); tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    confrows.append([fam,kind,mr.Cutoff,tn,fp,fn,tp,(tp+tn)/len(y)])
confusions=pd.DataFrame(confrows,columns=["Model_family","Algorithm","Cutoff","TN","FP","FN","TP","Accuracy"])

# Incremental paired bootstrap for LR models.
def incremental(y,p0,p1,B=1000,seed=SEED):
    y=np.asarray(y);p0=np.asarray(p0);p1=np.asarray(p1)
    def calc(ix):
        yy=y[ix];a=p0[ix];b=p1[ix]
        da=roc_auc_score(yy,b)-roc_auc_score(yy,a)
        idi=(b[yy==1].mean()-b[yy==0].mean())-(a[yy==1].mean()-a[yy==0].mean())
        nri=(np.mean(b[yy==1]>a[yy==1])-np.mean(b[yy==1]<a[yy==1]))+(np.mean(b[yy==0]<a[yy==0])-np.mean(b[yy==0]>a[yy==0]))
        return da,nri,idi
    main=calc(np.arange(len(y)));rng=np.random.default_rng(seed);i0=np.where(y==0)[0];i1=np.where(y==1)[0];vals=[]
    for _ in range(B): vals.append(calc(np.r_[rng.choice(i0,len(i0),True),rng.choice(i1,len(i1),True)]))
    v=np.asarray(vals);return main,np.quantile(v,[.025,.975],axis=0)
inc=[]
for label,base,add in [("Add FLI to Age+Sex","Age+Sex base","Age+Sex+FLI")]:
    main,ci=incremental(y,pred_store[(base,"LR")],pred_store[(add,"LR")],1000,SEED+999)
    inc.append([label,base,add,*main,ci[0,0],ci[1,0],ci[0,1],ci[1,1],ci[0,2],ci[1,2]])
increment=pd.DataFrame(inc,columns=["Comparison","Base_model","Extended_model","Delta_AUC","Continuous_NRI","IDI","Delta_AUC_L95","Delta_AUC_U95","NRI_L95","NRI_U95","IDI_L95","IDI_U95"])

# Persist out-of-fold probabilities for calibration plots, confusion matrices,
# and decision-curve verification in the revised manuscript.
predout=pd.DataFrame({"MetS_star":y})
for (fam,kind),pred in pred_store.items():
    predout[f"{fam}__{kind}"]=pred

# Decision-curve net benefit using OOF LR predictions.
dca=[]
for fam in ["Age+Sex base","Age+Sex+FLI","Enzymes only","HSI only"]:
    p=pred_store[(fam,"LR")]
    for t in np.arange(.05,.51,.05):
        pred=p>=t; tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));nb=tp/len(y)-fp/len(y)*t/(1-t)
        treat_all=y.mean()-(1-y.mean())*t/(1-t)
        dca.append([fam,round(t,2),nb,treat_all,0.0])
dca=pd.DataFrame(dca,columns=["Model_family","Threshold","Net_benefit","Treat_all","Treat_none"])

# Simple DeLong implementation for correlated AUCs.
def midrank(x):
    order=np.argsort(x); z=x[order]; n=len(x); T=np.zeros(n);i=0
    while i<n:
        j=i
        while j<n and z[j]==z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1;i=j
    out=np.empty(n);out[order]=T;return out
def delong_pair(y,s1,s2):
    y=np.asarray(y); order=np.argsort(-y); m=int(y.sum()); scores=np.vstack([s1,s2])[:,order]; n=len(y)-m
    tx=np.array([midrank(x[:m]) for x in scores]);ty=np.array([midrank(x[m:]) for x in scores]);tz=np.array([midrank(x) for x in scores])
    aucs=(tz[:,:m].sum(1)/m-(m+1)/2)/n
    v01=(tz[:,:m]-tx)/n;v10=1-(tz[:,m:]-ty)/m
    sx=np.atleast_2d(np.cov(v01));sy=np.atleast_2d(np.cov(v10));S=sx/m+sy/n
    var=S[0,0]+S[1,1]-2*S[0,1];z=(aucs[0]-aucs[1])/np.sqrt(max(var,1e-15));return aucs[0]-aucs[1],2*norm.sf(abs(z))
delrows=[]
for outcome in ["MetS_recalc","MetS_star"]:
 for subgroup,dd in [("Overall",d),("Female",d[d.Sex=="F"]),("Male",d[d.Sex=="M"])]:
  y0=dd[outcome].to_numpy(); alt=dd.ALT_U_L.to_numpy()
  for m in ["AST_U_L","GGT_U_L","FLI_recalc","HSI_recalc"]:
   diff,pv=delong_pair(y0,dd[m].to_numpy(),alt);delrows.append([outcome,subgroup,m,"ALT_U_L",diff,pv])
delong=pd.DataFrame(delrows,columns=["Outcome","Subgroup","Marker","Reference","AUC_difference","DeLong_p"])

# Save machine-readable intermediates for workbook builder.
tables={"audit":audit,"processed":raw,"descriptive":desc,"sex_descriptive":sexdesc,"correlation":corr.reset_index(names="Variable"),"adjusted_associations":associations,"roc":rocres,"interaction":inter,"models":modelres,
        "folds":foldres,"permutation":permres,"incremental":increment,"dca":dca,"delong":delong,
        "oof_predictions":predout,"confusion_matrices":confusions}
for name,df in tables.items(): df.to_csv(OUTDIR/f"{name}.csv",index=False)

# One Excel workbook containing every numerical output table.
with pd.ExcelWriter(OUTDIR/"all_results.xlsx",engine="openpyxl") as writer:
    for name,df in tables.items(): df.to_excel(writer,sheet_name=name[:31],index=False)

# Software environment for reproducibility.
versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
try:
    import scipy, sklearn
    versions.update({"scipy":scipy.__version__,"scikit-learn":sklearn.__version__})
except Exception: pass
(OUTDIR/"software_versions.json").write_text(json.dumps(versions,indent=2))

summary={
 "source_n":len(raw),"primary_n":len(d),"primary_mets_cases":int(d.MetS_recalc.sum()),"primary_mets_star_cases":int(d.MetS_star.sum()),
 "stored_cases":int(raw.Stored_MetS_binary.sum()),"status_disagreements":int(raw.Stored_status_mismatch.sum()),
 "best_model":modelres.sort_values("AUC",ascending=False).iloc[0].to_dict(),
 "fli_mets_auc":rocres.query("Outcome=='MetS_recalc' and Subgroup=='Overall' and Marker=='FLI_recalc'").iloc[0].to_dict(),
 "fli_star_auc":rocres.query("Outcome=='MetS_star' and Subgroup=='Overall' and Marker=='FLI_recalc'").iloc[0].to_dict(),
 "hsi_mets_auc":rocres.query("Outcome=='MetS_recalc' and Subgroup=='Overall' and Marker=='HSI_recalc'").iloc[0].to_dict(),
}
(OUTDIR/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
print(json.dumps(summary,indent=2,default=float))
