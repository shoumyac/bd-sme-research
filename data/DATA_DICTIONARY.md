# Data dictionary: `bangladesh_sme_1200_records.csv`

1,200 SME loan applications, Bangladesh, loans originated July 2023 to June 2025 (records taken at the first payment date after the first anniversary; collected July 2024 to June 2026). One row per application. No missing values.
Source: BASIC Bank Limited and Bangladesh Bank, obtained for this research with permission from the Ministry of Finance of Bangladesh. All columns, including the free-text descriptions and the outcome, were supplied by the banks. The research extract contains no personal identifiers.

Descriptions below follow the column names as supplied; where a definition is not documented in the source we say so rather than guess.

| Column | Type | Values / range | Used as | Notes |
|---|---|---|---|---|
| `business_id` | text | SME0001 to SME1200 | identifier | Not a feature |
| `sector` | category | Retail 353, Manufacturing 337, Services 305, Agriculture 205 | feature, subgroup | |
| `business_age_years` | integer | 1 to 19 | feature | |
| `legal_status` | category | Sole 623, Limited 327, Partnership 250 | feature | |
| `division` | category | Dhaka 405, Chittagong 234, Khulna 161, Rajshahi 123, Barisal 102, Rangpur 90, Sylhet 85 | feature, subgroup | Administrative division |
| `location_type` | category | Urban 936, Rural 264 | feature, subgroup | |
| `employees_count` | integer | 2 to 49 | feature | |
| `monthly_revenue_bdt` | integer | 34,577 to 3,567,055 | feature | BDT per month |
| `monthly_profit_bdt` | integer | 5,198 to 1,337,403 | feature | BDT per month |
| `existing_loans` | category | Yes 830, No 370 | feature | |
| `prior_loan_count` | integer | 0 to 3 | feature | |
| `late_payment_history` | category | Never 793, Rarely 217, Sometimes 190 | feature | |
| `loan_purpose` | category | Equipment 301, Working capital 240, Expansion 231, Renovation 226, Inventory 202 | feature | |
| `loan_amount_bdt` | integer | 58,338 to 5,000,000 (mean about 2.16 million) | feature | BDT |
| `preferred_tenure_months` | integer | 6 to 36 (mean 18.9) | feature | |
| `collateral_level` | category | High 431, Low 392, Moderate 377 | feature | |
| `cashflow_stability` | integer | 1 to 5 | feature | Ordinal score as supplied |
| `repayment_confidence` | integer | 1 to 5 | feature | Ordinal score as supplied |
| `owner_education` | category | Higher Secondary 295, Secondary 264, Bachelor's 248, Primary 229, Master's 164 | feature, subgroup | |
| `days_payable_outstanding` | integer | 7 to 90 | feature | |
| `business_description`, `loan_purpose_description`, `repayment_plan_description`, `risk_factors_description` | text | free text | **not used** by the federated experiments | Used only in the centralised benchmark (text features) |
| `missed_payment_last_12m` | category | Yes 303, No 897 | outcome | Identical to `target_missed_payment`. **Never use as a feature (it is the label).** |
| `target_missed_payment` | 0/1 | 1 = missed any scheduled payment in the 12 months after disbursement (303 of 1,200, 25.25%) | **label** | Wider than the Basel default (90 days past due) |
| `split` | category | train 960, test 240 | not used | Train/test split carried over from earlier work. The federated scripts make their own stratified 80/20 split (random_state 42) and do not read this column |

## Features used by the federated experiments

Ten numeric (`business_age_years`, `employees_count`, `monthly_revenue_bdt`, `monthly_profit_bdt`, `prior_loan_count`, `loan_amount_bdt`, `preferred_tenure_months`, `cashflow_stability`, `repayment_confidence`, `days_payable_outstanding`) and nine categorical (`sector`, `legal_status`, `division`, `location_type`, `existing_loans`, `late_payment_history`, `loan_purpose`, `collateral_level`, `owner_education`). After standardising the numeric columns and one-hot encoding the categorical ones there are 44 model inputs. The identifier, the four free-text columns, the duplicate label column `missed_payment_last_12m` and `split` are excluded. The lists are `NUM` and `CAT` in `federated/federated_xai_experiment_v2.py`.

## Reading the file

Plain CSV with a header row; the scripts read it with the pandas defaults.
