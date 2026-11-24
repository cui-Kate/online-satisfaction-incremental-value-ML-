import json
import platform
import importlib.metadata as md
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits
from study import DATA, RES, SEED, EXPECTED_SHA, FIELDS, PAIR, PRIMARY, candidates, load_data, tune, prob, write_json

def bootstrap_weights(y, repetitions=3000, seed=1102):
    rng = np.random.default_rng(seed)
    w = np.zeros((repetitions+1,len(y)),dtype=float)
    w[0] = 1
    for value in (0,1):
        ix = np.flatnonzero(y==value)
        w[1:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=repetitions)
    return w

def estimates(y, scores, weights):
    order = np.argsort(-scores,kind="stable")
    s = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)!=0)+1]
    pos = np.add.reduceat(weights[:,order]*y[order],starts,axis=1)
    neg = np.add.reduceat(weights[:,order]*(1-y[order]),starts,axis=1)
    cp,cn = pos.cumsum(axis=1),neg.cumsum(axis=1)
    p,n = cp[:,-1],cn[:,-1]
    precision = np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0)
    ap = (precision*pos).sum(axis=1)/p
    auc = (pos*(n[:,None]-cn+0.5*neg)).sum(axis=1)/(p*n)
    predicted = scores>=0.5
    tp = weights @ (y*predicted)
    tn = weights @ ((1-y)*(~predicted))
    fpos, fneg = n-tn, p-tp
    f1 = 0.5*(2*tp/(2*tp+fpos+fneg)+2*tn/(2*tn+fpos+fneg))
    metrics = {"AP":ap,"AUROC":auc,"macro_F1":f1,"accuracy":(tp+tn)/(p+n),
               "Brier":(weights@(scores-y)**2)/(p+n),"sensitivity":tp/p,"specificity":tn/n}
    reference=[average_precision_score(y,scores),roc_auc_score(y,scores),f1_score(y,predicted,average="macro"),accuracy_score(y,predicted),brier_score_loss(y,scores)]
    np.testing.assert_allclose([metrics[k][0] for k in list(metrics)[:5]],reference,atol=1e-12)
    return metrics

def summarize(samples):
    out={}
    for key,arr in samples.items():
        lower,upper=np.quantile(arr[1:],[.025,.975])
        out.update({key:arr[0],key+"_lo":lower,key+"_hi":upper})
    return out

def main():
    RES.mkdir(exist_ok=True)
    raw,df=load_data()
    y=(df.outcome=="Good").astype(int)
    train,test=train_test_split(df.index.to_numpy(),test_size=.20,random_state=SEED,stratify=y)
    split=pd.Series("development",index=df.index)
    split.loc[test]="holdout"
    pd.DataFrame({"source_row":df.index+1,"split":split}).to_csv(RES/"split_manifest.csv",index=False)
    binary=y.loc[test].to_numpy()
    weights=bootstrap_weights(binary)
    records=[]; comparisons=[]; grids=[]; holdout={}
    write_json("data_audit.json",{"source_sha256":EXPECTED_SHA,"rows":len(df),"columns":len(raw.columns),
        "raw_na_cells":int(raw.isna().sum().sum()),"duplicate_full_rows":int(raw.duplicated().sum()),
        "duplicate_predictor_rows":int(raw.iloc[:,:-1].duplicated().sum()),
        "labels":df.outcome.value_counts().to_dict(),"stage":df.stage.value_counts().to_dict(),
        "age_range":[df.age.min(),df.age.max()],"school_age_range":[df.loc[df.stage=="school","age"].min(),df.loc[df.stage=="school","age"].max()],
        "split_class_counts":pd.crosstab(split,df.outcome).to_dict(orient="index"),
        "performance_values":sorted(df.performance.unique().tolist())})
    pd.DataFrame(FIELDS,columns=["original_field","analysis_key","label","type"]).to_csv(RES/"data_dictionary.csv",index=False)
    counts=[]
    for key in PAIR:
        for score,sub in df.groupby(key):
            counts.append({"rating":key,"level":score,"n":len(sub),"Bad":(sub.outcome=="Bad").sum(),"Average":(sub.outcome=="Average").sum(),"Good":(sub.outcome=="Good").sum(),"Good_fraction":(sub.outcome=="Good").mean()})
    pd.DataFrame(counts).to_csv(RES/"rating_counts.csv",index=False)
    def fit(name,keys,tr,spec):
        result=tune(name,keys,df,y,tr)
        for i,params in enumerate(result.cv_results_["params"]):
            grids.append({"specification":spec,"model":name,"settings":json.dumps(params,sort_keys=True),"fold_mean_AP":result.cv_results_["mean_test_score"][i],
                          "fold_sd_AP":result.cv_results_["std_test_score"][i],"selected":i==result.best_index_})
        return result
    def score(name,tag,yt,s,wt,cv=np.nan,params=""):
        m=estimates(yt,s,wt)
        tn,fp,fn,tp=confusion_matrix(yt,s>=.5,labels=[0,1]).ravel()
        records.append({"specification":tag,"model":name,"n":len(yt),"Good":int(yt.sum()),"CV_AP":cv,"parameters":params,
                        **summarize(m),"TN":tn,"FP":fp,"FN":fn,"TP":tp})
        return m
    def contrast(tag,full,omit):
        comparisons.append({"contrast":tag,**summarize({k:full[k]-omit[k] for k in ("AP","AUROC","macro_F1","Brier")})})
    fitted={};primary_scores={};best_cv={}
    s0=np.full(len(test),y.loc[train].mean())
    score("Prevalence baseline","primary",binary,s0,weights)
    holdout["Prevalence baseline"]=s0
    for name in candidates():
        result=fit(name,PRIMARY,train,"primary")
        s=prob(result.best_estimator_,df.loc[test,PRIMARY])
        primary_scores[name]=score(name,"primary",binary,s,weights,result.best_score_,json.dumps(result.best_params_))
        best_cv[name]=result.best_score_
        fitted[name]=result.best_estimator_;holdout[name]=s
        print("fit",name,"development AP",round(result.best_score_,4),flush=True)
    winner=max(best_cv,key=best_cv.get)
    joblib.dump(fitted[winner],RES/"selected_model.joblib",compress=3)
    def lrvariant(keys,tr,te,tag,wt):
        result=fit("Logistic regression",keys,tr,tag)
        s=prob(result.best_estimator_,df.loc[te,keys]);yt=y.loc[te].to_numpy()
        if np.array_equal(te,test):holdout[tag]=s
        return score("Logistic regression",tag,yt,s,wt,result.best_score_,json.dumps(result.best_params_))
    reduced=[k for k in PRIMARY if k not in PAIR]
    m=lrvariant(reduced,train,test,"omit_pair",weights)
    contrast("Primary LR: retain minus omit both ratings",primary_scores["Logistic regression"],m)
    a=lrvariant(PRIMARY+["performance"],train,test,"with_performance",weights)
    b=lrvariant(reduced+["performance"],train,test,"with_performance_omit_pair",weights)
    contrast("LR: pair contrast conditional on performance",a,b)
    contrast("LR: add self-rated performance to primary",a,primary_scores["Logistic regression"])
    tr=train[(df.loc[train,"stage"]!="school").to_numpy()]
    te=test[(df.loc[test,"stage"]!="school").to_numpy()]
    wt=bootstrap_weights(y.loc[te].to_numpy())
    a=lrvariant(PRIMARY,tr,te,"postsecondary",wt)
    b=lrvariant(reduced,tr,te,"postsecondary_omit_pair",wt)
    contrast("Postsecondary LR: retain minus omit both ratings",a,b)
    cat=df.outcome.map({"Bad":0,"Average":1,"Good":2})
    tri=[]
    for label,feat in [("all_primary",PRIMARY),("omit_pair",reduced)]:
        mod=tune("Logistic regression",feat,df,cat,train,scoring="f1_macro")
        pred=mod.predict(df.loc[test,feat]);pr=mod.predict_proba(df.loc[test,feat])
        tri.append({"specification":label,"n":len(test),"macro_F1":f1_score(cat.loc[test],pred,average="macro"),
                    "macro_AUROC_ovr":roc_auc_score(cat.loc[test],pr,multi_class="ovr",average="macro"),
                    "Good_AP":average_precision_score(binary,pr[:,2]),"accuracy":accuracy_score(cat.loc[test],pred),
                    "train_CV_F1":mod.best_score_,"confusion_matrix_Bad_Average_Good":confusion_matrix(cat.loc[test],pred,labels=[0,1,2]).tolist()})
    write_json("three_class.json",tri)
    pd.DataFrame(records).to_csv(RES/"performance.csv",index=False)
    pd.DataFrame(comparisons).to_csv(RES/"paired_contrasts.csv",index=False)
    pd.DataFrame(grids).to_csv(RES/"cv_grid.csv",index=False)
    pd.DataFrame({"source_row":test+1,"outcome":binary,**holdout}).to_csv(RES/"holdout_predictions.csv",index=False)
    write_json("run.json",{"selected_by_development_AP":winner,"development_n":len(train),"holdout_n":len(test),
        "primary_model":"Logistic regression","primary_features":PRIMARY,"bootstrap_replicates":3000,"split_seed":SEED,"cv_seed":2026,
        "bootstrap_seed":1102,"python":platform.python_version(),"versions":{k:md.version(k) for k in ["numpy","scipy","pandas","scikit-learn","matplotlib","joblib","xgboost","threadpoolctl"]}})
    print("Selected",winner,flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()

