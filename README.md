# Grocery Sales Forecasting

Forecasting daily sales for 54 Ecuadorian grocery stores (Kaggle *Store Sales – Time Series Forecasting*, Corporación Favorita), comparing gradient-boosting models tuned with Optuna.

Built as a course project for **Data Management for Data Science (CS 639), UW–Madison, Spring 2026**, starting from a course-provided notebook structure.

## What I did

- **Data integration:** joined store transactions, national/regional holidays, and daily oil prices onto ~3M rows of sales data.
- **Feature engineering:** linearly interpolated missing oil prices, one-hot encoded categorical features, and added calendar features.
- **Exploratory analysis:** measured how transactions, oil price, and product family relate to sales (Spearman correlation, OLS), and looked at monthly and weekly seasonality.
- **Leak-free validation:** held out the final labeled block (2017-07-17 to 2017-07-31), right before the test period, so no future data leaks into training.
- **Modeling:** trained LightGBM, CatBoost, and XGBoost, then tuned each with Optuna (10 trials). Also fit a Ridge regression as a simple linear comparison.

## Results (validation RMSLE, lower is better)

| Model | Default settings | Tuned with Optuna |
| --- | --- | --- |
| XGBoost | 1.453 | **0.924** |
| LightGBM | 1.163 | 1.046 |
| CatBoost | 1.434 | 1.074 |

Tuning cut XGBoost's validation error by 36%, making it the best model after tuning.

## Repository

- `notebooks/sales_forecasting.ipynb` — the full pipeline, with plots.
- `requirements.txt` — Python packages used.

The dataset is not included; download it from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). The notebook also uses a small course-provided `helper` module (data loading and the RMSLE metric) that is not redistributed here.
