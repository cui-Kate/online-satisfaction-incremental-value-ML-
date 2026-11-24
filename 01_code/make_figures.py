from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve
from scipy.stats import norm
from study import RES, FIG, LABELS

BLUE = "#17647a"
GOLD = "#9e4b13"
INK = "#202b36"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.labelsize":10,
                     "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":.8,"pdf.fonttype":42,"savefig.facecolor":"white"})

def save(fig, stem):
    fig.savefig(FIG / (stem + ".png"), dpi=600, bbox_inches="tight",pad_inches=.09)
    fig.savefig(FIG / (stem + ".pdf"), bbox_inches="tight",pad_inches=.09)
    plt.close(fig)

def plot_label(value):
    return value.replace("Instructor help with questions", "Instructor help\nwith questions")

def rating_figure():
    data = pd.read_csv(RES/"rating_counts.csv")
    z = norm.ppf(.975)
    p, n = data.Good_fraction, data.n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    data["wilson_lower"] = center-radius
    data["wilson_upper"] = center+radius
    data.to_csv(RES/"figure1_coordinates.csv",index=False)
    fig,axes = plt.subplots(1,2,sharey=True,figsize=(6.85,2.70))
    for ax,key,color,marker in zip(axes,["interaction","clarification"],[BLUE,GOLD],["o","s"]):
        sub = data[data.rating==key]
        ax.errorbar(sub.level,sub.Good_fraction,yerr=[sub.Good_fraction-sub.wilson_lower,sub.wilson_upper-sub.Good_fraction],
                    fmt=marker+"-",capsize=3,color=color,lw=1.65,markersize=5)
        ax.set(xlim=(.75,5.25),ylim=(0,.87),xticks=range(1,6),yticks=np.arange(0,.81,.2),xlabel="Item rating")
        ax.set_title(LABELS[key],loc="left",fontsize=10.5,fontweight="bold")
        ax.grid(axis="y",color="#dce1e4",lw=.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].set_ylabel("Share selecting Good")
    fig.subplots_adjust(left=.12,right=.99,bottom=.21,top=.88,wspace=.15)
    save(fig,"item_ratings")

def precision_figure():
    scores = pd.read_csv(RES/"holdout_predictions.csv")
    summary = pd.read_csv(RES/"performance.csv")
    ys = scores.outcome
    fig,ax = plt.subplots(figsize=(6.45,3.02))
    rows=[]
    for pred,spec,name,color,dash in [
        ("Logistic regression","primary","Two ratings retained",BLUE,"-"),("omit_pair","omit_pair","Two ratings omitted",GOLD,"--")]:
        pr,rec,_ = precision_recall_curve(ys,scores[pred])
        ap = summary[(summary.specification==spec)&(summary.model=="Logistic regression")].AP.iloc[0]
        ax.step(rec,pr,where="post",lw=1.6,ls=dash,color=color,label=f"{name}  (AP {ap:.3f})")
        rows.extend({"specification":spec,"recall":r,"precision":p} for r,p in zip(rec,pr))
    ax.axhline(ys.mean(),color="#535c62",lw=1.0,ls=":",label=f"Holdout prevalence  ({ys.mean():.3f})")
    ax.set(xlim=(0,1.005),ylim=(0,1.03),xlabel="Recall",ylabel="Precision")
    ax.grid(axis="y",color="#dce1e4",lw=.6)
    ax.legend(loc="lower left",ncol=1,frameon=True,framealpha=.95,edgecolor="none",fontsize=8.1)
    fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.97)
    pd.DataFrame(rows).to_csv(RES/"figure2_coordinates.csv",index=False)
    save(fig,"precision_recall")

def attribution_figure():
    data = pd.read_csv(RES/"shap_global.csv").head(8).iloc[::-1].copy()
    fig,ax = plt.subplots(figsize=(6.55,3.45))
    y = np.arange(len(data))
    kwargs = {"linestyle":"none","markersize":5}
    ax.plot(data.holdout*100,y,"o",color=BLUE,label="Original holdout",**kwargs)
    ax.plot(data.development_73*100,y-.18,"^",color=GOLD,label="Development refit 1",**kwargs)
    ax.plot(data.development_137*100,y+.18,"s",mfc="white",color="#56616a",label="Development refit 2",**kwargs)
    ax.set(yticks=y,yticklabels=data.label.map(plot_label),xlim=(-.18,11),xticks=range(0,11,2),
           xlabel="Mean absolute contribution to P(Good), percentage points")
    ax.set_axisbelow(True)
    ax.grid(axis="x",color="#dce1e4",lw=.6)
    ax.legend(loc="lower right",fontsize=8,frameon=True,edgecolor="none",framealpha=.95)
    fig.subplots_adjust(left=.24,right=.99,bottom=.18,top=.97)
    save(fig,"model_attributions")

if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    rating_figure()
    precision_figure()
    attribution_figure()

