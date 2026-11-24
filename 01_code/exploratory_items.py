import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from study import RES, PAIR, PRIMARY, load_data, tune, prob
from fit_models import bootstrap_weights, estimates, summarize

def main():
    _,df=load_data()
    y=df.outcome.eq("Good").astype(int)
    split=pd.read_csv(RES/"split_manifest.csv")
    train=split.loc[split.split=="development","source_row"].to_numpy()-1
    pred=pd.read_csv(RES/"holdout_predictions.csv")
    test=pred.source_row.to_numpy()-1
    yt=y.loc[test].to_numpy()
    weights=bootstrap_weights(yt)
    base=[k for k in PRIMARY if k not in PAIR]
    scores={"both":pred["Logistic regression"].to_numpy(),"neither":pred["omit_pair"].to_numpy()}
    settings=[("interaction_only_added",base+["interaction"]),("clarification_only_added",base+["clarification"]),("only_two_items",PAIR)]
    for name,features in settings:
        fitted=tune("Logistic regression",features,df,y,train)
        scores[name]=prob(fitted.best_estimator_,df.loc[test,features])
    samples={k:estimates(yt,v,weights)["AP"] for k,v in scores.items()}
    contrasts=[("full_minus_interaction_only", "both","interaction_only_added"),
               ("full_minus_clarification_only","both","clarification_only_added")]
    all_rows=[{"comparison":name,**summarize({"AP":arr})} for name,arr in samples.items()]
    for label,f,g in contrasts:
        all_rows.append({"comparison":label,**summarize({"AP":samples[f]-samples[g]})})
    pd.DataFrame(all_rows).to_csv(RES/"exploratory_items.csv",index=False)
    pd.DataFrame({"source_row":test+1,**scores}).to_csv(RES/"exploratory_item_scores.csv",index=False)
    pooled=[]
    for seed in (510,511,512):
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        predictions={"full":np.empty(len(train)),"reduced":np.empty(len(train))}
        for train_local,valid_local in folds.split(train,y.loc[train]):
            for tag,feats in [("full",PRIMARY),("reduced",base)]:
                fitted=tune("Logistic regression",feats,df,y,train[train_local])
                predictions[tag][valid_local]=prob(fitted.best_estimator_,df.loc[train[valid_local],feats])
        aps={k:average_precision_score(y.loc[train],v) for k,v in predictions.items()}
        pooled.append({"outer_seed":seed,"n":len(train),"AP_full":aps["full"],"AP_reduced":aps["reduced"],"delta_AP":aps["full"]-aps["reduced"]})
    pd.DataFrame(pooled).to_csv(RES/"nested_development_AP.csv",index=False)

if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()

