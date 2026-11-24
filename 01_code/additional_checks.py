import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from study import RES, FIG, PAIR, FIELDS, LABELS, load_data, write_json
from make_figures import BLUE, GOLD, save, plot_label

def describe():
    raw,df=load_data()
    manifest=pd.read_csv(RES/"split_manifest.csv").set_index("source_row")
    hold=pd.read_csv(RES/"holdout_predictions.csv").set_index("source_row")
    predictors=df.drop(columns="outcome").to_numpy()
    pairs=[]
    for i in range(len(df)-1):
        near=np.flatnonzero((predictors[i+1:]!=predictors[i]).sum(axis=1)<=1)+i+1
        for j in near:
            pairs.append({"source_row_a":i+1,"source_row_b":j+1,
                          "split_a":manifest.loc[i+1,"split"],"split_b":manifest.loc[j+1,"split"],
                          "predictor_differences":int((predictors[i]!=predictors[j]).sum())})
    pd.DataFrame(pairs).to_csv(RES/"near_duplicate_audit.csv",index=False)
    omit=set()
    for pair in pairs:
        if pair["split_a"]!=pair["split_b"]:
            omit.add(pair["source_row_a"] if pair["split_a"]=="holdout" else pair["source_row_b"])
    checked=hold.drop(index=list(omit))
    a=average_precision_score(checked.outcome,checked["Logistic regression"])
    b=average_precision_score(checked.outcome,checked["omit_pair"])
    vars=PAIR+["performance","internet","grades"]
    rho=df[vars].corr(method="spearman")
    rho.to_csv(RES/"item_spearman.csv",index_label="item")
    write_json("additional_audit.json",{"age_median":df.age.median(),"age_quartiles":df.age.quantile([.25,.75]).tolist(),
              "near_pairs_on_22_fields":len(pairs),"holdout_rows_with_near_development_match":len(omit),
              "posthoc_near_match_exclusion_n":len(checked),"posthoc_AP_full":a,"posthoc_AP_reduced":b,"posthoc_delta":a-b})

def examples():
    tab=pd.read_csv(RES/"shap_examples.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.9),sharex=True)
    for ax,(quant,sub) in zip(axes,tab.groupby("score_quantile")):
        sub=sub.sort_values("rank")
        base=sub.base.iloc[0]; score=sub.score.iloc[0]
        amounts=list(sub.probability_contribution)
        amounts.append(score-base-sum(amounts))
        labels=[plot_label(LABELS[k]) for k in sub.feature]+["Other items"]
        curr=base
        for i,(label,dx) in enumerate(zip(labels,amounts)):
            ax.barh(i,abs(dx),left=min(curr,curr+dx),height=.57,color=BLUE if dx>=0 else GOLD)
            ax.plot([curr+dx]*2,[i+.3,i+.71],color="#687077",lw=.65)
            curr+=dx
        ax.set_yticks(range(5),labels,fontsize=8);ax.invert_yaxis()
        ax.axvline(base,color="#89939b",ls=":",lw=.85)
        ax.plot(score,4.63,marker="D",ms=4,color="#1b2734")
        ax.set_title(f"{int(quant*100)}th percentile   P(Good) = {score:.3f}",loc="left",fontsize=9.4,fontweight="bold")
        ax.set_xlim(0,.55);ax.set_xlabel("Predicted probability",fontsize=9)
        ax.tick_params(axis="x",labelsize=8)
    fig.subplots_adjust(left=.19,right=.99,bottom=.18,top=.85,wspace=.88)
    save(fig,"local_contributions")

if __name__=="__main__":
    describe()
    examples()

