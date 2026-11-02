# External datasets used for the generalisation check

`federated/external_datasets.py` repeats the explanation-drift audit on three public credit datasets. They are **not** stored in this repository. `federated/fetch_external_data.py` downloads them into `data/external/` (git-ignored) and checks the German file against the UCI original.

| Dataset | Source | Size used | Licence |
|---|---|---|---|
| Statlog (German Credit Data) | UCI Machine Learning Repository, id 144, https://doi.org/10.24432/C5NC77 (Hofmann, 1994) | 1,000 loans, 30% bad. Sex and personal status, and foreign-worker status are dropped | CC BY 4.0 |
| Default of credit card clients (Taiwan) | UCI, id 350, https://doi.org/10.24432/C55S3H (Yeh and Lien, 2009) | 30,000 clients, 5,000 sampled at random (seed 42). Sex is dropped | CC BY 4.0 |
| Taiwanese bankruptcy prediction | UCI, id 572, https://doi.org/10.24432/C5004D | 6,819 firms, 3.2% bankrupt, 94 ratios after dropping a constant column | CC BY 4.0 |

Reference: Yeh, I.-C. and Lien, C.-H. (2009), The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients, *Expert Systems with Applications* 36(2), 2473-2480.

Clients are stylised and built from attributes, not real institutions: German credit by loan purpose, card default by education and age, bankruptcy by terciles of the borrowing-dependency ratio. Preprocessing is fitted on the training rows only.
