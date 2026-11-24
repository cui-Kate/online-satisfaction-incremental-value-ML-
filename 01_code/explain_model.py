import json
import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from study import RES, PRIMARY, LABELS, load_data, prob, write_json

def attribute(pipe, frame, background_rows, target_rows, seed):
    transform = pipe.named_steps["transform"]
    bg = np.random.default_rng(seed).choice(background_rows, size=64, replace=False)
    background = transform.transform(frame.loc[bg, PRIMARY])
    target = transform.transform(frame.loc[target_rows, PRIMARY])
    explainer = shap.TreeExplainer(pipe.named_steps["model"], data=background,
                                   feature_perturbation="interventional", model_output="probability")
    values = np.asarray(explainer.shap_values(target, check_additivity=False))
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    groups = list(transform.transformers_[0][2])
    encoder = transform.named_transformers_["cat"]
    for key, levels in zip(transform.transformers_[1][2], encoder.categories_):
        groups.extend([key] * len(levels))
    if len(groups) != values.shape[1]:
        raise ValueError("Transformed columns and group labels disagree.")
    grouped = pd.DataFrame({key: values[:, np.asarray(groups) == key].sum(axis=1)
                            for key in PRIMARY}, index=target_rows)
    prediction = prob(pipe, frame.loc[target_rows, PRIMARY])
    residual = float(np.abs(base + grouped.sum(axis=1).to_numpy() - prediction).max())
    if residual > 1e-6:
        raise ValueError(f"Probability additivity error {residual}")
    return grouped, base, prediction, residual, bg

def main():
    _, frame = load_data()
    y = frame.outcome.eq("Good").astype(int)
    manifest = pd.read_csv(RES / "split_manifest.csv")
    dev = manifest.loc[manifest.split == "development", "source_row"].to_numpy() - 1
    test = pd.read_csv(RES / "holdout_predictions.csv").source_row.to_numpy() - 1
    selected = json.loads((RES/"run.json").read_text())["selected_by_development_AP"]
    if selected != "Gradient boosting":
        raise ValueError("This explanation was specified for the selected boosting estimator.")
    model = joblib.load(RES / "selected_model.joblib")
    phi, base, prediction, residual, bg = attribute(model, frame, dev, test, 901)
    individual = phi.assign(source_row=test+1, base_probability=base, model_probability=prediction)
    individual.to_csv(RES / "shap_by_row.csv", index=False)
    main_importance = phi.abs().mean()
    importance = pd.DataFrame({"holdout": main_importance})
    diagnostics = [{"sample":"holdout","n":len(test),"base_probability":base,"max_additivity_error":residual,
                    "background_rows":[int(i+1) for i in bg]}]
    for seed in (73, 137):
        inner_train, inner_test = train_test_split(dev, test_size=.25, random_state=seed, stratify=y.loc[dev])
        fitted = clone(model).fit(frame.loc[inner_train, PRIMARY], y.loc[inner_train])
        values, ref, _, err, bg_inner = attribute(fitted, frame, inner_train, inner_test, seed)
        item = "development_" + str(seed)
        importance[item] = values.abs().mean()
        diagnostics.append({"sample":item, "n":len(inner_test), "base_probability":ref,"max_additivity_error":err,
                            "rank_correlation_with_holdout":float(spearmanr(main_importance,importance[item]).statistic),
                            "background_rows":[int(i+1) for i in bg_inner]})
    importance.insert(0, "label", [LABELS[key] for key in importance.index])
    importance.index.name = "feature"
    importance.sort_values("holdout",ascending=False).to_csv(RES/"shap_global.csv")

    examples = []
    for q in (.1,.9):
        loc = int(np.argmin(np.abs(prediction - np.quantile(prediction, q))))
        idx = test[loc]
        keys = phi.loc[idx].abs().nlargest(4).index
        for rank,key in enumerate(keys,1):
            examples.append({"score_quantile":q, "source_row":idx+1,"score":prediction[loc],"base":base,
                             "observed":frame.at[idx,"outcome"],"rank":rank,"feature":key,
                             "source_value":frame.at[idx,key],"probability_contribution":phi.at[idx,key]})
    pd.DataFrame(examples).to_csv(RES/"shap_examples.csv",index=False)
    write_json("shap_diagnostics.json",{"shap_version":shap.__version__,"model":selected,
              "units":"probability of Good relative to each development background","runs":diagnostics})

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

