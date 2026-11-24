# Quantitative excerpt of the local analysis specification

Extracted from the project specification recorded before resampling or scoring. The quantitative sections below are transcribed from the internal record. This record was not publicly preregistered.

**Question.**Do the respondent's online interaction and instructor clarification ratings jointly add information for classifying a Good overall online-education rating in the pandemic survey?

**Primary estimand.** Paired difference in held-out average precision (AP) for two L2 logistic regression models, one retaining the two ratings, the other omitting them. Use the 21 non-outcome features other than concurrent self-rated online performance in the full specification, and the remaining 19 in the reduced model. Compare the same participants at natural class prevalence. Report paired stratified percentile bootstrap 95% intervals. Same-row explanatory classification, not prospective screening.

**Validation.** Original published CSV with SHA-256 verification. Stratified random 80/20 holdout, seed 42, row as observation. No row or subject identifier is provided. Fit encoders, scalers, hyperparameters and any model selection in training only. No synthetic observations. Five shuffled stratified folds within development, seed 2026. Tune by mean AP. No threshold optimization on holdout. Use 0.50 as displayed threshold. ROC AUC, macro-F1, Brier and confusion matrix supplement AP. Prevalence and majority-class baselines.

**Comparisons and sensitivity.** Apply the same holdout to all seven original estimator families. Highest training-CV AP selects the exploratory benchmark model for attribution; the primary contrast remains logistic. Reintroduce concurrent performance, repeat the pair comparison among postsecondary survey records, and fit an additional three-category logistic sensitivity. Paired comparisons are descriptive, one planned primary contrast, no multiple significance claims.

**Explanations.** Shapley attributions of positive-class probability computed over whole original features. Use development backgrounds and verify per-row additivity. Contrast classification with thematic interpretation; no intervention effect or individual triage claim. Assess stability using further splits of the development set, without consulting the primary holdout.
