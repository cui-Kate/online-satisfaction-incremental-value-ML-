# Results by research question

All names in the table are relative to 03_results unless indicated. Row numbers refer to positions in the original public survey, starting at 1 below the header. Holdout labels describe survey records and model-estimated scores are on the Good-class probability scale.

| Evidence in the article | Files | How to check it |
|---|---|---|
| Original survey and its licence | 02_data/online_education_review.csv, 02_data/SOURCE.md | Check the depositor, version and original SHA-256 before analysis. |
| Coding and allocation | data_dictionary.csv, data_audit.json, split_manifest.csv, run.json | Inspect 23 fields, binary outcome coding, 826/207 split, seeds and versions. |
| Item frequencies, uncertainty and association (Figure 2) | rating_counts.csv, figure1_coordinates.csv, item_spearman.csv | Reconstruct the item-rating plot and descriptive associations. |
| Integrity and near matches | near_duplicate_audit.csv, additional_audit.json | Inspect near matches and the additional source audit. |
| Fitting and benchmark | cv_grid.csv, performance.csv, selected_model.joblib | Consult development settings, holdout performance, confusion counts and the fitted estimator selected for explanations. |
| Primary and performance-adjusted increments | holdout_predictions.csv, paired_contrasts.csv, comparator_contrast.csv | Verify paired estimates and the explicitly post hoc difference between increments. |
| Precision–recall curve (Figure 3) | figure2_coordinates.csv, holdout_predictions.csv | Reconstruct the curve and AP from the same holdout scores. |
| Alternative items and development validation | exploratory_items.csv, exploratory_item_scores.csv, nested_development_AP.csv | Distinguish single-item holdout estimates and pooled out-of-fold development estimates. |
| Original three labels | three_class.json | Check class order, confusion counts and multiclass performance. |
| Boosting explanations (Figures 4 and 5) | shap_global.csv, shap_by_row.csv, shap_examples.csv, shap_diagnostics.json | Inspect signed row contributions, aggregated rankings, the two score-quantile examples and reconstruction checks. |
| Study design (Figure 1) | 06_method_figure/ | The supplied draft, editable LaTeX source and final vector and raster files document the methods diagram. |
| Four empirical figures (Figures 2–5) | 04_figures/item_ratings, precision_recall, model_attributions, local_contributions (each as PDF and PNG) | Each filename matches its corresponding PDF asset in the manuscript. |
| Analysis decisions | 05_specifications/primary_quantitative_excerpt.md, exploratory_checks.md, comparator_contrast_specification.md | The first file identifies the provenance of its excerpt. The last two document exploratory checks. |
