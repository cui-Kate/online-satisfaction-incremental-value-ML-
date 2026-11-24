# Reproducing the online learning satisfaction analysis

**Article.** The Incremental Value of Student Ratings of Online Interaction and Instructor Help with Questions for Classifying Online Learning Satisfaction.

## Repository and dataset attribution

Project repository: https://github.com/cui-Kate/online-satisfaction-incremental-value-ML-

This repository accompanies the article above and contains 53 files in total: six directories of data, code and resources, plus this README and RESULTS_INDEX.md. The original survey is Sujatha Radhakrishnan (2021), Online Education System - Review, version 1, Mendeley Data, https://doi.org/10.17632/bzk9zbyvv7.1. The unmodified CSV is redistributed under CC BY 4.0; see 02_data/SOURCE.md. The original dataset DOI identifies the survey, not this analysis code or its derived results.

The source data licence applies to the original survey. No separate software licence is assigned in this release.

## Environment and execution

The computations were run with Python 3.10.16 on Windows and one numerical thread. From the extracted archive root, install the pinned top-level packages into a Python 3.10 environment and run the files in order. All paths are relative to this directory. No keys, remote model APIs or notebook state are needed.

~~~text
python -m pip install -r 01_code/requirements.txt
python 01_code/fit_models.py
python 01_code/explain_model.py
python 01_code/make_figures.py
python 01_code/additional_checks.py
python 01_code/exploratory_items.py
python 01_code/compare_increments.py
python 01_code/validate_results.py
~~~

On Windows PowerShell, set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 before running the commands to match the single-thread reference run. Outputs already included in this release can be regenerated in a fresh copy containing only 01_code and 02_data.

Fit models first to create the split, out-of-sample scores and selected estimator. Run explain_model.py before generating the figures. The file study.py supplies the shared input and model definitions. The model is trained locally and saved as 03_results/selected_model.joblib for use by the explanation and validation scripts. The source dataset must match its published SHA-256 digest. Full numerical reproduction should use the pinned libraries. Small floating point differences across platforms may occur.

## Data and output files

- 02_data/online_education_review.csv is the unmodified published survey with a shorter filename. See 02_data/SOURCE.md for attribution and licensing. The CSV has 1,033 data rows. Source row numbers in outputs start at 1, excluding the header. They identify public file rows, not individual students.
- 03_results/data_dictionary.csv maps exact original headings to analysis keys. 03_results/data_audit.json gives category totals and input quality checks.
- split_manifest.csv records all original split assignments. holdout_predictions.csv contains binary labels and probabilities for the primary benchmark and full-sample logistic variants. Subgroup scores are summarized in performance.csv.
- performance.csv contains class counts, probabilities-based measures, threshold metrics, 95% intervals, confusion counts and selected hyperparameters. paired_contrasts.csv uses identical bootstrap replicates for each pair. cv_grid.csv gives the development selection history. Rows labelled primary in performance.csv include seven families and the constant reference.
- three_class.json contains macro measures and original-label confusion matrices. exploratory_items.csv, exploratory_item_scores.csv and nested_development_AP.csv report the exploratory item and repeated development-fold analyses.
- shap_global.csv summarizes grouped attributions in probability units. shap_by_row.csv contains the 21 grouped contributions, base value and probability for every original holdout row. shap_examples.csv identifies the examples. shap_diagnostics.json records background rows and additivity checks.
- rating_counts.csv, item_spearman.csv, near_duplicate_audit.csv and additional_audit.json document descriptive and similarity checks. figure1_coordinates.csv and figure2_coordinates.csv contain plotting coordinates. The grouped-attribution and local-contribution plots are drawn from the SHAP CSV files. Output image filenames describe their content and remain stable when figure numbering changes.
- 04_figures contains 600 dpi PNG images and vector PDFs. PDF creation dates may differ when regenerated.

The primary quantity is the full-minus-reduced difference in average precision. The positive class is Good. The test fraction is 50/207. Confidence intervals are fixed-model, class-stratified record bootstraps. The exploratory specification does not change the primary comparison. Resampling cannot establish representativeness, causal effects or identity independence.

The additional post hoc contrast in comparator_contrast.csv applies the same class-stratified weights to all four stored probability rankings. The script compare_increments.py independently verifies all 3,000 contrast values by repeating the sampled rows explicitly.

## Release contents

RESULTS_INDEX.md maps manuscript results to their machine-readable source files. The original public survey is redistributed under CC BY 4.0 as documented in 02_data/SOURCE.md. The source data licence applies to the survey. Regenerate the fitted estimator from the distributed code if platform versions differ. The empirical figures are provided both as vector PDFs and RGB-compatible PNGs. Figure 1 describes the study design. The folder 06_method_figure includes its source draft, editable typesetting, vector PDF and publication-resolution PNG. The empirical figures in 04_figures are Figures 2–5.


## Terminology alignment on 19 November 2026

The manuscript describes the focal items as student ratings of online interaction and instructor help with questions. The internal key clarification retains its original spelling for reproducibility; its display label is Instructor help with questions. The unmodified source field is Clearing doubts with faculties in online mode. Dated specification excerpts and the original diagram draft preserve their original wording. Plot labels and the current data dictionary use the revised wording.
