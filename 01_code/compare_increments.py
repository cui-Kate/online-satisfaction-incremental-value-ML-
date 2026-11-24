import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fit_models import bootstrap_weights, estimates
from study import RES


def main():
    predictions = pd.read_csv(RES / "holdout_predictions.csv")
    labels = predictions["outcome"].to_numpy()
    fields = ["Logistic regression", "omit_pair", "with_performance", "with_performance_omit_pair"]
    weights = bootstrap_weights(labels)
    aps = [estimates(labels, predictions[key].to_numpy(), weights)["AP"] for key in fields]
    without_performance = aps[0] - aps[1]
    with_performance = aps[2] - aps[3]
    change = without_performance - with_performance
    intervals = np.quantile(change[1:], [0.025, 0.975])
    verified = []
    scores = predictions[fields].to_numpy()
    for sample in weights[1:]:
        rows = np.repeat(np.arange(len(labels)), sample.astype(int))
        values = [average_precision_score(labels[rows], scores[rows, i]) for i in range(4)]
        verified.append((values[0] - values[1]) - (values[2] - values[3]))
    np.testing.assert_allclose(change[1:], verified, atol=1e-12, rtol=0)
    summary = {"n": len(labels), "positive": int(labels.sum()),
               "delta_AP_without_performance": without_performance[0],
               "delta_AP_with_performance": with_performance[0], "difference": change[0],
               "lower_95": intervals[0], "upper_95": intervals[1], "replicates": len(change)-1}
    pd.DataFrame([summary]).to_csv(RES / "comparator_contrast.csv", index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

