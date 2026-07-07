# Cambodia Used Car Residual Value Estimator

Interactive Streamlit research prototype for estimating used-car residual value in Cambodia.

## Important Reference-Date Note

This app is **not a live market-price service**.

- **Base data reference period:** 2024 Q3 observed market-price data
- **2024 output:** observed-reference model estimate anchored to the 2024 Q3 dataset
- **2025-2027 outputs:** projected values based on 2024 Q3 data and data-driven depreciation assumptions

For example, if the user selects **2026 projection**, the app estimates a projected 2026 value by starting from the 2024 Q3 model value and applying an annual depreciation factor. It should not be described as a real-time June 2026 market price.

## Features

- Vehicle-level value estimator
- Valuation reference selector: 2024 Q3, 2025 projection, 2026 projection, 2027 projection
- 2024 Q3 base value and projected value display
- Annual depreciation assumption and source explanation
- Estimated price range
- Implied residual value ratio using reference new-car price
- Residual risk view using residual ratio and model-grade data
- Market dashboard by brand, segment, age, and residual ratio
- Sold-vehicle benchmark table
- Methodology and limitations page

## Data Files

```text
app.py
train_model.py
requirements.txt
data/
  market_price_clean.csv
  new_price_reference.csv
  sold_vehicle_validation.csv
  grade_lookup.csv
model/
  residual_price_model.pkl
  model_metrics.json
```

## How to Run Locally

From inside the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How to Deploy on Streamlit Community Cloud

1. Upload the project files to a GitHub repository.
2. Make sure `app.py` and `requirements.txt` appear at the repository root.
3. Open Streamlit Community Cloud.
4. Connect the GitHub repository.
5. Select `app.py` as the main file.
6. Deploy.

## Model Summary

The base model uses a Random Forest regression model trained on `log(car_price)`.
The model estimates a 2024 Q3 reference market value first. Future values are generated using a depreciation projection layer.

## Limitations

- No mileage variable
- No accident history
- No detailed vehicle-condition variable
- No service-record variable
- No real-time market feed
- Future-year outputs are projections, not observed prices
- Macroeconomic shifts, exchange rates, tax changes, import rules, and inventory shocks after 2024 Q3 are not directly modeled

## Suggested Portfolio Description

Built an interactive Python/Streamlit research prototype that converts Cambodia used-car market data into a residual value estimator. The app distinguishes between 2024 Q3 observed-reference values and 2025-2027 projected values, using a depreciation-based projection layer to avoid presenting estimates as real-time market prices.
