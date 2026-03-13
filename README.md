# AI-Driven Credit Risk Assessment for Bangladesh SMEs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18525278.svg)](https://doi.org/10.5281/zenodo.18525278)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Data: CC-BY 4.0](https://img.shields.io/badge/Data-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This repository contains the dataset and complete analysis pipeline for:

> **Structured Data Suffices: Machine Learning for SME Credit Risk Assessment in an Emerging Market**
>
> Submitted to *Financial Innovation* (Springer Nature)

**Key Findings:**
- **0.900 ROC-AUC** with simple logistic regression on structured features (5-fold CV)
- Text features (TF-IDF from loan narratives) add **<1% value** — structured data alone suffices
- **Payment history** is the dominant predictor (coefficient ±1.9)
- Train–test gap of only **0.017**, confirming strong generalisation
- Simple models outperform XGBoost (0.875) and Random Forest (0.810)

## Repository Structure

```
.
├── data/
│   └── bangladesh_sme_1200_records.csv   # 1,200 SME loan applications
├── notebooks/
│   └── bd_sme_1200_complete_analysis.ipynb   # Complete analysis pipeline
├── results/                              # Generated outputs
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── cv_results_summary.csv
│   ├── feature_importance_lr.csv
│   ├── feature_importance_lr.png
│   ├── model_comparison.png
│   ├── pr_curves_all_models.png
│   ├── roc_curves_all_models.png
│   ├── shap_bar.png
│   ├── shap_summary.png
│   └── target_distribution.png
├── CITATION.cff
├── LICENSE                               # MIT License
├── README.md
└── requirements.txt
```

## Dataset

**File:** `data/bangladesh_sme_1200_records.csv`

1,200 SME loan applications from Bangladesh (2022–2023) comprising:

| Feature type | Count | Examples |
|---|---|---|
| Numeric | 8 | monthly revenue, profit, loan amount, cashflow stability |
| Categorical | 11 | sector, division, late payment history, collateral level |
| Text | 4 | business description, loan purpose, repayment plan, risk factors |
| Target | 1 | missed_payment_last_12m (binary) |

**Class distribution:** 897 non-default (74.8%) / 303 default (25.2%)

**Ethics:** All data anonymised. No personally identifiable information retained. Written informed consent obtained by partner financial institutions at time of loan application.

**Data licence:** CC-BY 4.0

## Reproduction

### Requirements

- Python 3.8+
- Jupyter Notebook

### Setup

```bash
git clone https://github.com/shoumyac/bd-sme-research.git
cd bd-sme-research
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook notebooks/bd_sme_1200_complete_analysis.ipynb
# Run all cells — expected runtime: ~5–10 minutes
```

## Results

### Model Performance (5-Fold Stratified Cross-Validation)

| Model | ROC-AUC | PR-AUC | Brier Score | Train–Test Gap |
|---|---|---|---|---|
| **LR (Tabular only)** | **0.900 ± 0.024** | **0.776 ± 0.045** | 0.194 | **0.017** |
| LR (Tabular + Text) | 0.899 ± 0.023 | 0.780 ± 0.044 | 0.191 | 0.029 |
| XGBoost | 0.875 ± 0.019 | 0.717 ± 0.035 | 0.174 | 0.125 |
| Random Forest | 0.810 ± 0.019 | 0.611 ± 0.038 | 0.208 | 0.190 |

### Top Predictive Features (Logistic Regression Coefficients)

| Rank | Feature | Coefficient |
|---|---|---|
| 1 | late_payment_history (Sometimes) | +1.944 |
| 2 | late_payment_history (Never) | −1.909 |
| 3 | loan_amount_bdt | +1.741 |
| 4 | cashflow_stability | −1.190 |
| 5 | collateral_level (Low) | +1.112 |

## Figures

All figures are in `results/` at 300 DPI:

| Figure | Description |
|---|---|
| `roc_curves_all_models.png` | ROC curves for all models |
| `pr_curves_all_models.png` | Precision–Recall curves |
| `calibration_curve.png` | Probability calibration |
| `shap_summary.png` | SHAP beeswarm plot |
| `shap_bar.png` | Mean absolute SHAP importance |
| `feature_importance_lr.png` | Top 20 LR coefficients |
| `confusion_matrix.png` | Best model confusion matrix |
| `model_comparison.png` | CV performance comparison |
| `target_distribution.png` | Class balance |

## Citation

```bibtex
@article{chowdhury2026structured,
  title={Structured Data Suffices: Machine Learning for {SME} Credit Risk Assessment in an Emerging Market},
  author={Das, Anmita and Paul, Sushanta and Chowdhury, Shoumya},
  journal={Financial Innovation},
  year={2026},
  note={Under review},
  doi={10.5281/zenodo.18525278}
}
```

## Licence

- **Code:** MIT License
- **Data:** Creative Commons Attribution 4.0 International (CC-BY 4.0)

## Authors

- **Anmita Das** — University of Melbourne
- **Sushanta Paul** — Bangladesh Customs, National Board of Revenue
- **Shoumya Chowdhury** *(corresponding)* — University of Melbourne — shoumyac@student.unimelb.edu.au

## Links

- **Zenodo:** [https://doi.org/10.5281/zenodo.18525278](https://doi.org/10.5281/zenodo.18525278)
- **Journal:** [Financial Innovation (Springer Nature)](https://link.springer.com/journal/40854)
