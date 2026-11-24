import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score
from study import RES, FIG, DATA, PRIMARY, PAIR, load_data, prob

def main():
    raw,frame=load_data()
    y=frame.outcome.eq("Good").astype(int)
    split=pd.read_csv(RES/"split_manifest.csv")
    assert len(split)==len(raw)==1033 and split.source_row.is_unique
    assert set(split.source_row)==set(range(1,len(raw)+1))
    out=pd.read_csv(RES/"holdout_predictions.csv").set_index("source_row")
    dev=set(split.loc[split.split=="development","source_row"])
    assert set(out.index)==set(split.loc[split.split=="holdout","source_row"]) and not dev.intersection(out.index)
    np.testing.assert_array_equal(out.outcome.to_numpy(),y.iloc[out.index-1])
    test=out.outcome.to_numpy()
    metrics=pd.read_csv(RES/"performance.csv")
    for _,r in metrics[metrics.specification.isin(["primary","omit_pair","with_performance","with_performance_omit_pair"])].iterrows():
        key=r.model if r.specification=="primary" else r.specification
        ss=out[key].to_numpy()
        expected=(average_precision_score(test,ss),roc_auc_score(test,ss),f1_score(test,ss>=.5,average="macro"),np.mean((ss-test)**2))
        np.testing.assert_allclose(r[["AP","AUROC","macro_F1","Brier"]].to_numpy(dtype=float),expected,atol=1e-12)
        np.testing.assert_array_equal(confusion_matrix(test,ss>=.5,labels=[0,1]).ravel(),r[["TN","FP","FN","TP"]].to_numpy(dtype=int))
    pos,neg=np.flatnonzero(test),np.flatnonzero(1-test)
    c=pd.read_csv(RES/"paired_contrasts.csv").iloc[0]
    full,reduced=out["Logistic regression"].to_numpy(),out["omit_pair"].to_numpy()

    rng=np.random.default_rng(1102)
    buf=np.zeros((3000,len(test)),dtype=int)
    for ix in (neg,pos):
        buf[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=3000)
    delta=[]
    for weights in buf:
        draw=np.repeat(np.arange(len(test)),weights)
        delta.append(average_precision_score(test[draw],full[draw])-average_precision_score(test[draw],reduced[draw]))
    np.testing.assert_allclose(np.quantile(delta,[.025,.975]),[c.AP_lo,c.AP_hi],atol=1e-12)
    fit=joblib.load(RES/"selected_model.joblib")
    run=json.loads((RES/"run.json").read_text())
    np.testing.assert_allclose(prob(fit,frame.iloc[out.index-1][PRIMARY]),out[run["selected_by_development_AP"]],atol=1e-12)
    shap=pd.read_csv(RES/"shap_by_row.csv").set_index("source_row")
    np.testing.assert_allclose(shap[PRIMARY].sum(axis=1)+shap.base_probability,shap.model_probability,atol=1e-7)
    np.testing.assert_allclose(shap.loc[out.index,"model_probability"],out["Gradient boosting"],atol=1e-12)
    for spec in json.loads((RES/"three_class.json").read_text()):
        mat=np.asarray(spec["confusion_matrix_Bad_Average_Good"])
        np.testing.assert_array_equal(mat.sum(axis=1),[50,107,50])
        assert abs(np.mean(2*np.diag(mat)/(mat.sum(axis=1)+mat.sum(axis=0)))-spec["macro_F1"])<1e-12
    for key in PAIR:
        tab=pd.read_csv(RES/"rating_counts.csv").query("rating==@key")
        assert tab.n.sum()==1033 and tab.Good.sum()==251
    for stem in ["item_ratings","precision_recall","model_attributions","local_contributions"]:
        assert all((FIG/(stem+ext)).stat().st_size>5000 for ext in [".pdf",".png"])
    print(f"PASS | rows {len(raw)} | development {len(dev)} | holdout {len(out)} | Good {sum(test)}")
    print(f"PASS | paired AP {c.AP:.6f} | 95% interval [{c.AP_lo:.6f}, {c.AP_hi:.6f}] | 3000 explicit row bootstraps")
    print("PASS | binary scores, confusion counts, three-class matrices, positive-class contributions, and four figures")

if __name__=="__main__":
    main()

