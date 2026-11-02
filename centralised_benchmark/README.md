# Centralised benchmark

The first analysis of the dataset: a regularised logistic regression, a random forest and XGBoost on the structured columns, with and without TF-IDF features from the four free-text columns, evaluated by stratified 5-fold cross-validation, with SHAP explanations.

Notebook: `notebooks/bd_sme_1200_complete_analysis.ipynb` (about 5 to 10 minutes). Install `requirements.txt` in this folder first. The notebook reads `../data/bangladesh_sme_1200_records.csv`, so start Jupyter from the `notebooks/` folder.

Stored results (`results/cv_results_summary.csv`, mean +/- standard deviation over folds):

| Model | Test ROC-AUC | Test PR-AUC | Train ROC-AUC |
|---|---|---|---|
| Logistic regression, structured only | 0.8995 +/- 0.0240 | 0.7758 +/- 0.0445 | 0.9167 |
| Logistic regression, structured + text | 0.8987 +/- 0.0232 | 0.7801 +/- 0.0443 | 0.9280 |
| Random forest | 0.8103 +/- 0.0185 | 0.6110 +/- 0.0376 | 1.0000 |
| XGBoost | 0.8751 +/- 0.0186 | 0.7173 +/- 0.0352 | 1.0000 |

Adding text features does not change discrimination. Figures are in `results/`.
