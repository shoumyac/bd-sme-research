# AI-Driven Credit Risk Assessment for Bangladesh SMEs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the complete analysis pipeline for "AI-Driven Credit Risk Assessment for Bangladesh SMEs: Achieving 90% Accuracy with Structured Data Alone", submitted to Financial Innovation journal.

**Key Results:**
- 🎯 **0.900 ROC-AUC** (5-fold cross-validation)
- 📊 Simple logistic regression outperforms complex models
- 📝 Text features add <1% value (unnecessary)
- 🔍 Payment history is the dominant predictor

## Repository Structure

```
.
├── data/                              # Dataset
│   └── bangladesh_sme_1200_records.csv    # 1200 SME loan applications
├── notebooks/                         # Analysis notebooks
│   └── bd_sme_1200_complete_analysis.ipynb    # Main analysis
├── results/                           # Generated outputs
│   ├── *.png                              # Visualizations (9 figures)
│   ├── cv_results_summary.csv             # Performance metrics
│   └── feature_importance_lr.csv          # Feature importance rankings
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
└── environment.yml                    # Conda environment (optional)
```

## Dataset

**File:** `data/bangladesh_sme_1200_records.csv`

**Description:** 1,200 SME loan applications from Bangladesh (2022-2023) with:
- 10 numeric features (revenue, profit, loan amount, etc.)
- 9 categorical features (sector, division, payment history, etc.)
- 4 text features (business descriptions, loan narratives)
- 1 target variable (missed_payment_last_12m)

**Ethics:** All data anonymized. No personally identifiable information. Informed consent obtained by partner financial institutions.

**License:** CC-BY 4.0

## Reproduction Instructions

### Requirements

- Python 3.8+
- Jupyter Notebook
- See `requirements.txt` for full package list

### Installation

```bash
# Clone repository
git clone https://github.com/shoumyac/BD-SME-Credit-Risk.git
cd BD-SME-Credit-Risk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/bd_sme_1200_complete_analysis.ipynb
# Run all cells (Cell > Run All)
```

**Expected Runtime:** ~5-10 minutes on standard laptop

## Key Results

### Model Performance (5-Fold Cross-Validation)

| Model | Test ROC-AUC | Test PR-AUC | Brier Score | Train-Test Gap |
|-------|--------------|-------------|-------------|----------------|
| **Logistic Regression (Tabular Only)** | **0.900 ± 0.024** | **0.776 ± 0.045** | **0.194 ± 0.032** | **0.017** |
| Logistic Regression (Tab + Text) | 0.899 ± 0.023 | 0.780 ± 0.044 | 0.191 ± 0.026 | 0.029 |
| XGBoost | 0.875 ± 0.019 | 0.717 ± 0.035 | 0.174 ± 0.015 | 0.125 |
| Random Forest | 0.810 ± 0.019 | 0.611 ± 0.038 | 0.208 ± 0.007 | 0.190 |

### Top 5 Most Important Features

1. **late_payment_history** (±1.9) - Payment behavior dominates
2. **loan_amount_bdt** (+1.74) - Larger loans increase risk
3. **cashflow_stability** (-1.19) - Stable cashflow reduces risk
4. **collateral_level** (±1.1) - Low collateral increases risk
5. **monthly_profit_bdt** (-0.98) - Higher profit reduces risk

## Figures

All figures are publication-ready (300 DPI) and located in `results/`:

1. **target_distribution.png** - Class balance
2. **model_comparison.png** - Cross-validation performance
3. **confusion_matrix.png** - Classification results
4. **roc_curves_all_models.png** - ROC curves for all models
5. **pr_curves_all_models.png** - Precision-Recall curves
6. **calibration_curve.png** - Probability calibration
7. **feature_importance_lr.png** - Top 20 features
8. **shap_summary.png** - SHAP beeswarm plot
9. **shap_bar.png** - SHAP feature importance

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{chowdhury2026sme,
  title={AI-Driven Credit Risk Assessment for Bangladesh SMEs: Achieving 90\% Accuracy with Structured Data Alone},
  author={Chowdhury, Shoumya and Das, Anmita and Paul, Sushanta},
  journal={Financial Innovation},
  year={2026},
  note={Under review}
}
```

**Zenodo DOI:** https://doi.org/10.5281/zenodo.XXXXXXX (to be updated)

## License

- **Code:** MIT License (see LICENSE file)
- **Data:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Paper:** Copyright © 2026 Authors. Submitted to Financial Innovation.

## Authors

- **Shoumya Chowdhury** (Corresponding Author) - University of Melbourne - shoumyac@student.unimelb.edu.au
- **Anmita Das** - University of Melbourne - aadas@student.unimelb.edu.au
- **Sushanta Paul** - Bangladesh Customs, National Board of Revenue - sushanta.researcher@gmail.com

## Acknowledgments

We thank the financial institutions and microfinance organizations in Bangladesh that facilitated data collection, and the SME owners who participated in the survey.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: shoumyac@student.unimelb.edu.au

## Related Links

- **Paper:** [Link to be added after publication]
- **Zenodo Dataset:** https://doi.org/10.5281/zenodo.XXXXXXX
- **Financial Innovation Journal:** https://jfin-swufe.springeropen.com/

---

**Last Updated:** February 2026  
**Version:** 1.0.0  
**Status:** Under Review
