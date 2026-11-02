# Bangladesh SME Credit Risk: Dataset and Federated Explanation Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18525277.svg)](https://doi.org/10.5281/zenodo.18525277)
[![Code: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

A public dataset of 1,200 Bangladeshi SME loans and the code that studies it in two ways:

1. **Centralised benchmark** (`centralised_benchmark/`): a notebook comparing classifiers and explanation methods on the pooled data.
2. **Federated explanation analysis** (`federated/`): simulated federations of institutions, and a test of whether federated training changes the reasons a declined applicant would be given, compared with training on the pooled data.

The DOI above is the concept DOI and always resolves to the latest archived version.

## The data

| | |
|---|---|
| File | `data/bangladesh_sme_1200_records.csv` (1,200 rows, 27 columns, no missing values) |
| Source | BASIC Bank Limited and Bangladesh Bank, obtained for research with the permission of the Ministry of Finance of Bangladesh |
| Loans originated | July 2023 to June 2025 |
| Collected | July 2024 to June 2026. Each record was taken at the first payment date after the loan's first anniversary |
| Label | `target_missed_payment`: 1 if the SME missed any scheduled payment in the 12 months after disbursement (303 of 1,200, 25.25%) |
| Privacy | The research extract contains no personal identifiers or contact details |
| Licence | CC BY 4.0 |

Every column is described, with value counts and ranges, in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md). The label is wider than the Basel default definition (90 days past due), and the file has no rejected applicants.

## Repository map

```
.
├── README.md
├── CITATION.cff, .zenodo.json         citation and archive metadata
├── LICENSE (MIT, code), LICENSE-DATA (CC BY 4.0, data)
├── requirements.txt                   federated experiments
├── data/
│   ├── bangladesh_sme_1200_records.csv
│   └── DATA_DICTIONARY.md
├── centralised_benchmark/             first analysis of the data (notebook, figures, own requirements)
├── federated/                         federated experiments
│   ├── run_all.sh                     reproduces everything in about 10 minutes
│   ├── federated_xai_experiment.py        horizontal FL, Dirichlet stress test, fairness
│   ├── federated_xai_experiment_v2.py     models, partitions, aggregators; model comparison, client scaling, DP noise
│   ├── federated_xai_experiment_v3.py     vertical and personalised federation
│   ├── mitigation.py                      shared-baseline explanations and the dispersion proposition
│   ├── explanation_drift.py               agreement with the pooled-data reference and the retraining noise floor
│   ├── summarise_drift.py                 summary tables
│   ├── cv_auc.py, cv_extra.py             5-fold cross-validated ROC-AUC
│   └── results_*/                         every output file (CSV, JSON, PNG)
└── docs/METHODS.md                    definitions and design of the explanation-drift analysis
```

## Quick start

```bash
git clone https://github.com/shoumyac/bd-sme-research.git
cd bd-sme-research
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
bash federated/run_all.sh
```

All randomness is seeded. The results stored in this repository were produced with numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1 and Python 3.13. The federated and explanation-drift results reproduce exactly. The tree-based classifiers in the ten-model comparison (`results_federated_v2/model_zoo.csv`) can differ by up to about 0.01 ROC-AUC across scikit-learn versions.

## What the federated analysis asks

A federated model is not expected to explain a decision exactly as an isolated model would, so a raw difference between explanations is not a failure. The question here is narrower: **does federated training change the reasons for a decline, relative to a model trained on the pooled data, by more than ordinary retraining changes them?**

* Clients are simulated by splitting the data into three institution archetypes (low, medium and high heterogeneity) and, separately, by Dirichlet label skew. All records come from one bank and the central bank, so the archetypes are stylised, not real institutions.
* Explanations are exact linear SHAP values for logistic regression and Integrated Gradients for a one-hidden-layer network.
* The comparison is made only for applicants the pooled-data reference model declines (score at least 0.30), because only those applicants receive an adverse-action notice.
* Agreement is the overlap (Jaccard) of the top five cited reasons, the share of cases with the same top reason, and the cosine similarity of the attribution vectors.
* A retraining noise floor (50 bootstrap retrainings of the pooled model) shows how much explanations move under routine retraining.

Full definitions are in [`docs/METHODS.md`](docs/METHODS.md). A selection of the verified results, for logistic regression under the high-heterogeneity partition (mean top-5 overlap with the pooled reference; the retraining noise floor is 0.829 with a 5th percentile of 0.763):

| Method | Top-5 overlap | Same top reason |
|---|---|---|
| FedAvg | 0.900 | 0.972 |
| FedProx | 0.895 | 0.967 |
| SCAFFOLD | 0.855 | 0.915 |
| Coordinate-wise median | 0.738 | 0.843 |
| Isolated local models | 0.782 | 0.839 |

All tables, including the multilayer perceptron, client-count, differential-privacy, label-skew and decline-threshold sweeps, are in `federated/results_drift/summary_*.csv`.

## Citing

Please cite the archived dataset and code (see `CITATION.cff`, or use GitHub's "Cite this repository"). Manuscripts that use this repository are in preparation and will be linked here when public.

## Authors

* Shoumya Chowdhury (corresponding), University of Melbourne, shoumyac@student.unimelb.edu.au
* Anmita Das, University of Melbourne
* Sushanta Paul, Bangladesh Customs, National Board of Revenue

## Licence

Code: MIT (`LICENSE`). Data and documentation: CC BY 4.0 (`LICENSE-DATA`).
