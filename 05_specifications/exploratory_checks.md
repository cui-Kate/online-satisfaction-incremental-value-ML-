# Additional explanatory checks

After the primary holdout comparison, three secondary analyses were specified to interpret the paired result. They are exploratory and do not redefine the primary comparison or model. Date: 16 November 2026.

1. Fit the same tuned logistic regression with each of the two ratings separately in addition to the 19 comparator features, and on the two ratings alone. Use the original split and development-only tuning. Compute paired fixed-holdout bootstrap intervals for adding each item after the other. These checks share the same holdout as the primary comparison.
2. Within the original 826 development records, repeat a nested five-fold stratified cross-validation three times. Tune each logistic model by five-fold AP in the inner development fold. Compute the full-minus-reduced difference in AP on each repeat's pooled out-of-fold predictions. The repeated results share data and are a stability description, not independent replications or a confidence interval.
3. Describe raw Spearman item correlations and inspect similarity on all 22 recorded predictors. Check the influence of any holdout record differing by at most one field from a development record. The file has no respondent identifiers.

