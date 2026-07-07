# Data Dictionary

## `market_price_clean.csv`

Main cleaned market-price dataset used for model training and dashboarding.

Key columns:

- `brand`: Vehicle brand
- `model_name`: Vehicle model name
- `model_norm`: Normalized model name
- `product_type`: New Car or Used Car
- `car_year`: Vehicle model year
- `vehicle_age`: Vehicle age as of the 2024 Q3 reference period
- `car_price`: Observed or listed market price in USD, based on the 2024 Q3 reference dataset
- `brand_origin`: Brand-origin category
- `segment`: Vehicle segment/body type
- `brand_tier`: Brand-tier grade where available
- `rv_grade`: Residual-value grade where available
- `risk_grade`: Risk grade where available
- `profit_grade`: Profitability grade where available
- `reference_new_price`: Model-level reference new-car price where available
- `brand_reference_new_price`: Brand-level fallback reference new-car price
- `implied_residual_ratio`: `car_price / reference_new_price` where available

## `new_price_reference.csv`

Reference new-car price lookup used to calculate implied residual ratios.

Key columns:

- `brand`
- `model_name`
- `reference_new_price`
- `new_price_observations`
- `brand_reference_new_price`
- `brand_new_obs`
- `brand_origin`

## `sold_vehicle_validation.csv`

Cleaned sold-vehicle benchmark list used for sanity-checking model output.

Key columns:

- `Type`
- `brand`
- `model_name`
- `car_year`
- `segment`
- `book_car_price`
- `sold_price`
- `sold_ratio_to_book`
- `brand_origin`
- `vehicle_age_at_sale_basis_2024`

## `grade_lookup.csv`

Vehicle-grade lookup table.

Key columns:

- `brand`
- `model_mid`
- `model_detail`
- `segment`
- `brand_tier`
- `rv_grade`
- `risk_grade`
- `profit_grade`
- `brand_delinquency_rate`
- `model_delinquency_rate`
- `model_profit_rate`

## Reference-Date Interpretation

The app's training data and base estimate are anchored to **2024 Q3 observed market-price data**.

- 2024 Q3 output = observed-reference estimate
- 2025-2027 outputs = projected values based on 2024 Q3 data and depreciation assumptions

The app does **not** provide a live June 2026 market quote.
