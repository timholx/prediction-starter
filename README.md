# Prediction Starter Guide

A hands-on tour of the models and habits that matter for predicting things from tabular and time-series data, from OLS to gradient boosting, ensembling and model interpretation.

Each notebook is short, self-contained and runnable top to bottom. Each one gives a brief explanation of the method, working examples on real or simulated data, and the takeaways worth remembering. Outputs are saved, so you can read them on GitHub without running anything.

## Start here: reference notebooks

| Notebook | What it's for |
|---|---|
| [**Quickstart pipeline**](notebooks/quickstart_pipeline.ipynb) | a short end-to-end pipeline for (financial) time series to run straight away: quick checks → intro plots → target → past-only features → walk-forward CV → predict-zero / Ridge / LightGBM → prediction sizing → test → slide summary |
| [Setup & installation](notebooks/setup_and_installation.ipynb) | installing Python, Jupyter, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna and SHAP (venv, conda, Colab); troubleshooting; a check cell that imports and fits each library |
| [pandas cheatsheet](notebooks/pandas_cheatsheet.ipynb) | runnable reference: reading, selecting, cleaning, strings, dates, group-by, merges, reshaping, leak-free feature engineering, quick plots, gotchas (pandas 3) |
| [Code templates](notebooks/code_templates.ipynb) | copy-paste functions: line/bar/hist/scatter/box/heatmap/small-multiples plots, model diagnostics (pred vs actual, residuals, deciles, importance, ROC/PR, learning curves, CV comparison), metrics, `compare_models`, timer, save/load |

## Notebooks

| # | Notebook | What you'll learn | Data |
|---|---|---|---|
| 00 | [Dataset interview playbook](notebooks/00_dataset_interview_playbook.ipynb) | a reusable, config-driven template: data-quality report, leakage scan, baseline → linear → boosting on the same folds, error analysis, presentation outline | California housing with planted problems |
| 01 | [Workflow, validation & leakage](notebooks/01_workflow_validation_and_leakage.ipynb) | baselines, CV splitters (KFold / Stratified / Group / TimeSeries), two leakage demos, choosing a metric | California housing, simulated |
| 02 | [Linear regression: OLS, Ridge, Lasso](notebooks/02_linear_regression_ols_ridge_lasso.ipynb) | statsmodels OLS, regularisation paths, p ≈ n, splines, shrinking weak predictions | diabetes, simulated |
| 03 | [Logistic regression & classification](notebooks/03_logistic_regression_and_classification.ipynb) | probabilities vs labels, ROC/PR, calibration, cost-based thresholds, class imbalance | breast cancer, wine |
| 04 | [KNN & SVM](notebooks/04_knn_and_svm.ipynb) | why scaling matters, choosing *k*, kernels, `C` × `gamma`, curse of dimensionality | wine, two moons |
| 05 | [Decision trees & random forests](notebooks/05_decision_trees_and_random_forests.ipynb) | depth vs overfitting, OOB score, impurity vs permutation importance, no extrapolation | California housing |
| 06 | [Gradient boosting & XGBoost](notebooks/06_gradient_boosting_and_xgboost.ipynb) | boosting from scratch, early stopping, key parameters, native categoricals | California housing, adult income |
| 07 | [LightGBM](notebooks/07_lightgbm.ipynb) | leaf-wise trees, `num_leaves`, `lgb.cv`, categoricals, monotone constraints | adult income, California housing |
| 08 | [CatBoost](notebooks/08_catboost.ipynb) | ordered target statistics, high-cardinality categoricals, target-encoding leakage | adult income, simulated |
| 09 | [Model comparison & tuning](notebooks/09_model_comparison_and_tuning.ipynb) | every model on the same folds, Optuna tuning, selection bias | California housing |
| 10 | [Ensembling & stacking](notebooks/10_ensembling_and_stacking.ipynb) | out-of-fold predictions, error correlation, averaging, NNLS weights, stacking | California housing |
| 11 | [Time series & weak signals](notebooks/11_time_series_and_weak_signals.ipynb) | past-only features, walk-forward CV with a gap, linear vs boosting, sizing predictions | simulated lead-lag market |
| 12 | [Model interpretation](notebooks/12_model_interpretation.ipynb) | permutation importance, PDP/ICE, SHAP | California housing |

Suggested path: **01 → 02 → 03 → 05 → 06/07/08 → 09 → 10**. Read 11 for anything time-ordered, and 12 whenever you need to explain a model.

### Preparing for a timed dataset interview
Focus on **00 → 01 → 02 → 11 (if time-ordered) → 07 → 12**. Notebook 00 is the template to run on the day. The others supply the reasoning you'll be asked to defend: validation choices, leakage, why a simple model is (or isn't) enough, and what drives the predictions. Tuning (09) and stacking (10) are worth a sentence in "next steps", not hours on the day.

## Which model should I try?

| Situation | Start with | Then try |
|---|---|---|
| Tabular data, mixed feature types, >1k rows | **LightGBM** (fast) or **CatBoost** (best defaults) | XGBoost, tuning, ensembling |
| Many / high-cardinality categorical features | **CatBoost** | LightGBM + out-of-fold target encoding |
| Small data (< ~1k rows) or need for interpretability | **Ridge / logistic regression** | splines, random forest, SVM |
| Very noisy target (returns, sports, A/B effects) | **Ridge** with walk-forward CV | regularised boosting; shrink predictions |
| Many more features than rows (p ≫ n) | **Ridge / Lasso / Elastic Net** | SVM |
| Need to extrapolate beyond the training range | **linear models** | trees on de-trended targets |
| Need calibrated probabilities | **logistic regression** | boosting + `CalibratedClassifierCV` |
| A robust "first look" with no tuning | **random forest** | — |

## Ten habits that matter more than the model

1. **Fix the validation scheme first**, and make it match how the model will be used: new rows, new groups (`GroupKFold`), or the future (walk-forward).
2. **Beat a baseline** (mean, majority class, predict zero, last value) before anything else.
3. **Put every learned preprocessing step inside the CV loop.** That includes scaling, imputation, feature selection and target encoding.
4. **Report the spread across folds**, not just the mean. Differences inside the noise aren't real.
5. **Features beat tuning.** Domain knowledge (groups, ratios, time since an event) is where the big gains are.
6. **Match the loss to the metric** (MSE → mean, MAE → median, log loss → probabilities).
7. **Early-stop boosting** on validation data; don't hand-pick `n_estimators`.
8. **Ensemble diverse models** using out-of-fold predictions.
9. **For weak signals, size matters**: check the out-of-sample slope of actual on predicted, and shrink when it's below 1.
10. **Explain the model to debug it.** A suspiciously important feature is often a leak.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Tested with Python 3.13, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, CatBoost 1.2 and Optuna 5.1.
The datasets come from scikit-learn (`load_*` ship with the library, while `fetch_california_housing` and `fetch_openml("adult")` download on first use and are cached) or are simulated in the notebook. No data files need to be committed.

Most notebooks run in under a minute on a laptop. Notebooks 09 (comparison + Optuna) and 10 (stacking) take 2–3 minutes each.

## Further reading

- *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani (free online)
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman (free online)
- scikit-learn user guide: [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- Docs: [XGBoost](https://xgboost.readthedocs.io/) · [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) · [CatBoost](https://catboost.ai/docs/) · [Optuna](https://optuna.readthedocs.io/) · [SHAP](https://shap.readthedocs.io/)
- *Advances in Financial Machine Learning* — López de Prado (purged/embargoed CV for overlapping labels)
