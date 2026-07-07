import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "model"

BASE_REFERENCE_YEAR = 2024
BASE_REFERENCE_PERIOD = "2024 Q3 observed market reference"
VALUATION_OPTIONS = {
    "2024 Q3 observed market reference": 2024,
    "2025 projection based on 2024 Q3 data": 2025,
    "2026 projection based on 2024 Q3 data": 2026,
    "2027 projection based on 2024 Q3 data": 2027,
}

st.set_page_config(
    page_title="Cambodia Used Car Residual Value Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    market = pd.read_csv(DATA_DIR / "market_price_clean.csv")
    new_ref = pd.read_csv(DATA_DIR / "new_price_reference.csv")
    sold = pd.read_csv(DATA_DIR / "sold_vehicle_validation.csv")
    grades = pd.read_csv(DATA_DIR / "grade_lookup.csv")
    with open(MODEL_DIR / "model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return market, new_ref, sold, grades, metrics

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "residual_price_model.pkl")

market, new_ref, sold, grades, metrics = load_data()
model = load_model()

FEATURE_COLS = [
    "brand",
    "model_name",
    "product_type",
    "car_year",
    "vehicle_age",
    "brand_origin",
    "segment",
    "brand_tier",
    "rv_grade",
    "risk_grade",
    "profit_grade",
]

# ----------------------------
# Helper functions
# ----------------------------
def currency(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"${float(x):,.0f}"


def pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x) * 100:.1f}%"


def robust_annual_depreciation_rate(df: pd.DataFrame):
    """Estimate a cross-sectional annual depreciation rate from observed used-car prices.

    The app uses 2024 Q3 observed prices as the base. For 2025-2027 projections,
    it applies a conservative annual depreciation factor estimated from median
    price changes by vehicle age. This is not a real-time market index.
    """
    if df is None or len(df) < 30:
        return None
    tmp = df[(df["product_type"] == "Used Car") & (df["car_price"] > 0)].copy()
    tmp = tmp[tmp["vehicle_age"].between(0, 20)]
    if len(tmp) < 30:
        return None

    med = tmp.groupby("vehicle_age")["car_price"].median().sort_index()
    rates = []
    for age in med.index:
        next_age = age + 1
        if next_age in med.index:
            p0 = med.loc[age]
            p1 = med.loc[next_age]
            if p0 > 0 and p1 > 0:
                rate = 1 - (p1 / p0)
                if 0.02 <= rate <= 0.30:
                    rates.append(rate)
    if not rates:
        return None
    return float(np.median(rates))


def get_projection_rate(market_df, brand, model_name, brand_origin, segment):
    """Fallback hierarchy for future-value projection.

    1. Same brand + model.
    2. Same segment + brand origin.
    3. Same segment.
    4. Overall used-car market.
    5. Conservative default.
    """
    candidates = [
        (
            market_df[(market_df["brand"] == brand) & (market_df["model_name"] == model_name)],
            "same brand-model depreciation curve",
        ),
        (
            market_df[(market_df["segment"] == segment) & (market_df["brand_origin"] == brand_origin)],
            "same segment and brand-origin depreciation curve",
        ),
        (
            market_df[market_df["segment"] == segment],
            "same segment depreciation curve",
        ),
        (
            market_df,
            "overall used-car market depreciation curve",
        ),
    ]
    for df, source in candidates:
        rate = robust_annual_depreciation_rate(df)
        if rate is not None:
            # Keep the projection conservative and avoid extreme extrapolation.
            return float(np.clip(rate, 0.03, 0.18)), source
    return 0.08, "conservative default annual depreciation assumption"


def risk_label(ratio, risk_grade):
    if ratio is None or np.isnan(ratio):
        return "Reference price needed"
    grade_text = str(risk_grade).upper()
    if ratio >= 0.65 and grade_text in {"A", "B"}:
        return "Low residual risk"
    if ratio >= 0.50:
        return "Moderate residual risk"
    return "High residual risk"


def mode_or_unknown(df, col):
    vals = df[col].dropna()
    vals = vals[vals.astype(str).str.lower() != "nan"]
    if len(vals) == 0:
        return "Unknown"
    modes = vals.mode()
    return modes.iloc[0] if len(modes) else vals.iloc[0]

# ----------------------------
# App header
# ----------------------------
st.title("Cambodia Used Car Residual Value Estimator")
st.caption(
    "Research prototype using cleaned Cambodia market-price, sold-vehicle, and model-grade datasets. "
    "The base model is anchored to 2024 Q3 observed market prices; 2025-2027 values are projections, not live market quotes."
)

st.info(
    "Data reference period: 2024 Q3 observed market data. "
    "When you select 2025, 2026, or 2027, the app projects forward from the 2024 Q3 base using a data-driven depreciation assumption. "
    "It is not a real-time 2026 market-price feed."
)

with st.expander("Important limitation", expanded=False):
    st.write(
        """
        This is a student research prototype, not a commercial valuation tool.
        The model does not include mileage, accident history, service record, ownership history,
        or detailed vehicle condition. Contract/customer identifiers were excluded from the app data.
        Projected values for 2025-2027 are scenario estimates based on 2024 Q3 price patterns and depreciation assumptions.
        """
    )

# ----------------------------
# Sidebar input
# ----------------------------
st.sidebar.header("Vehicle Input")

valuation_label = st.sidebar.selectbox(
    "Valuation Reference",
    list(VALUATION_OPTIONS.keys()),
    index=list(VALUATION_OPTIONS.keys()).index("2026 projection based on 2024 Q3 data"),
    help="2024 is the observed data reference. 2025-2027 are projected values based on depreciation assumptions.",
)
valuation_year = VALUATION_OPTIONS[valuation_label]
projection_years = max(0, valuation_year - BASE_REFERENCE_YEAR)

brands = sorted(market["brand"].dropna().unique())
brand = st.sidebar.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)

brand_models = sorted(market.loc[market["brand"] == brand, "model_name"].dropna().unique())
model_name = st.sidebar.selectbox("Model", brand_models)

product_types = sorted(market["product_type"].dropna().unique())
default_product_idx = product_types.index("Used Car") if "Used Car" in product_types else 0
product_type = st.sidebar.selectbox("Product Type", product_types, index=default_product_idx)

brand_model_rows = market[(market["brand"] == brand) & (market["model_name"] == model_name)]
min_year = int(max(2001, market["car_year"].min()))
max_year = int(min(BASE_REFERENCE_YEAR, market["car_year"].max()))
default_year = int(brand_model_rows["car_year"].median()) if len(brand_model_rows) else 2018
default_year = min(max(default_year, min_year), max_year)
car_year = st.sidebar.slider("Car Year", min_value=min_year, max_value=max_year, value=default_year)

base_vehicle_age = BASE_REFERENCE_YEAR - car_year
valuation_vehicle_age = valuation_year - car_year

# Default lookup values from same brand-model, else fallback
same = market[(market["brand"] == brand) & (market["model_name"] == model_name)]
brand_origin = mode_or_unknown(same, "brand_origin")
segment = mode_or_unknown(same, "segment")
brand_tier = mode_or_unknown(same, "brand_tier")
rv_grade = mode_or_unknown(same, "rv_grade")
risk_grade = mode_or_unknown(same, "risk_grade")
profit_grade = mode_or_unknown(same, "profit_grade")

# User can override price reference
ref_rows = new_ref[(new_ref["brand"] == brand) & (new_ref["model_name"] == model_name)]
if len(ref_rows):
    default_new_price = float(ref_rows["reference_new_price"].median())
else:
    brand_ref_rows = new_ref[new_ref["brand"] == brand]
    default_new_price = float(brand_ref_rows["brand_reference_new_price"].median()) if len(brand_ref_rows) else 0.0

use_reference = st.sidebar.checkbox("Use reference new price for residual ratio", value=True)
manual_new_price = st.sidebar.number_input(
    "Reference / Original New Car Price (USD)",
    min_value=0,
    max_value=300000,
    value=int(default_new_price) if default_new_price and not np.isnan(default_new_price) else 0,
    step=1000,
    disabled=not use_reference,
)

# ----------------------------
# Prediction and projection
# ----------------------------
# The model is trained on 2024 Q3 observations. Keep its age feature anchored to 2024.
input_df = pd.DataFrame([{
    "brand": brand,
    "model_name": model_name,
    "product_type": product_type,
    "car_year": car_year,
    "vehicle_age": base_vehicle_age,
    "brand_origin": brand_origin,
    "segment": segment,
    "brand_tier": str(brand_tier),
    "rv_grade": str(rv_grade),
    "risk_grade": str(risk_grade),
    "profit_grade": str(profit_grade),
}])

pred_log = model.predict(input_df[FEATURE_COLS])[0]
base_pred_price = float(np.expm1(pred_log))

annual_depr_rate, depr_source = get_projection_rate(market, brand, model_name, brand_origin, segment)
projection_factor = (1 - annual_depr_rate) ** projection_years
pred_price = base_pred_price * projection_factor

# Conservative uncertainty using model median error, comparable spread, and projection horizon
median_ae = float(metrics.get("median_ae_price_usd", 2500))
similar = market[
    (market["brand"] == brand)
    & (market["model_name"] == model_name)
    & (market["product_type"] == product_type)
    & (market["car_year"].between(car_year - 2, car_year + 2))
]
if len(similar) >= 3:
    q25, q75 = similar["car_price"].quantile([0.25, 0.75]).tolist()
    base_low = min(base_pred_price - median_ae, q25)
    base_high = max(base_pred_price + median_ae, q75)
else:
    base_low = base_pred_price - median_ae
    base_high = base_pred_price + median_ae

# Wider range for later-year projections because uncertainty compounds.
projection_uncertainty_buffer = median_ae * 0.25 * projection_years
low = max(0, base_low * projection_factor - projection_uncertainty_buffer)
high = max(0, base_high * projection_factor + projection_uncertainty_buffer)

if use_reference and manual_new_price > 0:
    residual_ratio = pred_price / manual_new_price
    low_ratio = low / manual_new_price
    high_ratio = high / manual_new_price
else:
    residual_ratio = None
    low_ratio = None
    high_ratio = None

# ----------------------------
# KPI cards
# ----------------------------
st.subheader(f"Estimated Value — {valuation_label}")
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Estimated {valuation_year} Value", currency(pred_price))
col2.metric("Estimated Range", f"{currency(low)} ~ {currency(high)}")
if residual_ratio is not None:
    col3.metric("Implied Residual Ratio", pct(residual_ratio))
else:
    col3.metric("Implied Residual Ratio", "N/A")
col4.metric("Risk View", risk_label(residual_ratio, risk_grade))

col5, col6, col7, col8 = st.columns(4)
col5.metric("2024 Q3 Base Value", currency(base_pred_price))
col6.metric("Projection Years", f"{projection_years} year(s)")
col7.metric("Annual Depreciation Assumption", pct(annual_depr_rate))
col8.metric("Vehicle Age at Valuation", f"{valuation_vehicle_age} years")

st.caption(
    f"Projection method: 2024 Q3 base model value × (1 - annual depreciation rate)^{projection_years}. "
    f"Depreciation source: {depr_source}."
)

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Estimator Detail",
    "Market Dashboard",
    "Sold Vehicle Benchmarks",
    "Methodology",
])

with tab1:
    st.subheader("Selected Vehicle Profile")

    left, right = st.columns([1.1, 1])
    with left:
        profile = pd.DataFrame({
            "Field": [
                "Brand", "Model", "Product Type", "Car Year",
                "Base Reference Period", "Base Vehicle Age",
                "Selected Valuation Year", "Vehicle Age at Valuation",
                "Brand Origin", "Segment / Body Type", "Brand Tier",
                "Residual Grade", "Risk Grade", "Profitability Grade",
                "Depreciation Source", "Annual Depreciation Assumption",
            ],
            "Value": [
                brand, model_name, product_type, car_year,
                BASE_REFERENCE_PERIOD, base_vehicle_age,
                valuation_year, valuation_vehicle_age,
                brand_origin, segment, brand_tier,
                rv_grade, risk_grade, profit_grade,
                depr_source, pct(annual_depr_rate),
            ],
        })
        st.dataframe(profile, hide_index=True, use_container_width=True)

    with right:
        st.write("Comparable observations")
        comp = market[
            (market["brand"] == brand)
            & (market["model_name"] == model_name)
            & (market["product_type"] == product_type)
        ].copy()
        if len(comp):
            comp_summary = pd.DataFrame({
                "Metric": ["Count", "Median 2024 Q3 Price", "Min Price", "Max Price", "Median Car Year"],
                "Value": [
                    f"{len(comp):,.0f}",
                    currency(comp["car_price"].median()),
                    currency(comp["car_price"].min()),
                    currency(comp["car_price"].max()),
                    f"{comp['car_year'].median():.0f}",
                ]
            })
            st.dataframe(comp_summary, hide_index=True, use_container_width=True)
        else:
            st.info("No exact comparable observations in the cleaned dataset.")

    projection_table = pd.DataFrame({
        "Valuation Year": [2024, 2025, 2026, 2027],
        "Vehicle Age": [year - car_year for year in [2024, 2025, 2026, 2027]],
        "Projected Value": [base_pred_price * ((1 - annual_depr_rate) ** max(0, year - BASE_REFERENCE_YEAR)) for year in [2024, 2025, 2026, 2027]],
    })
    projection_table["Projected Value"] = projection_table["Projected Value"].map(currency)
    st.write("Projection path from 2024 Q3 base value")
    st.dataframe(projection_table, hide_index=True, use_container_width=True)

    if len(same):
        st.write("Observed 2024 Q3 price pattern for selected model")
        history = same.groupby("car_year", as_index=False)["car_price"].median().sort_values("car_year")
        fig, ax = plt.subplots()
        ax.plot(history["car_year"], history["car_price"], marker="o")
        ax.set_xlabel("Car Year")
        ax.set_ylabel("Median observed price (USD)")
        ax.set_title(f"{brand} {model_name}: observed median price by car year")
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Market Dashboard")
    st.caption("All observed charts use the 2024 Q3 market-price dataset. Projection charts are derived from this reference base.")

    c1, c2 = st.columns(2)
    with c1:
        top_brands = (
            market[market["product_type"] == product_type]
            .groupby("brand", as_index=False)
            .agg(median_price=("car_price", "median"), observations=("car_price", "size"))
            .query("observations >= 20")
            .sort_values("median_price", ascending=False)
            .head(15)
        )
        st.write(f"Top median prices by brand — {product_type}, 2024 Q3 reference")
        st.bar_chart(top_brands.set_index("brand")["median_price"])

    with c2:
        seg = (
            market[market["product_type"] == product_type]
            .groupby("segment", as_index=False)
            .agg(median_price=("car_price", "median"), observations=("car_price", "size"))
            .query("observations >= 20")
            .sort_values("median_price", ascending=False)
            .head(15)
        )
        st.write(f"Median prices by segment — {product_type}, 2024 Q3 reference")
        st.bar_chart(seg.set_index("segment")["median_price"])

    c3, c4 = st.columns(2)
    with c3:
        st.write("Observed used-car depreciation pattern by vehicle age")
        used = market[(market["product_type"] == "Used Car") & (market["vehicle_age"].between(0, 20))]
        dep = used.groupby("vehicle_age", as_index=False)["car_price"].median().sort_values("vehicle_age")
        if len(dep):
            fig, ax = plt.subplots()
            ax.plot(dep["vehicle_age"], dep["car_price"], marker="o")
            ax.set_xlabel("Vehicle age at 2024 Q3")
            ax.set_ylabel("Median observed price (USD)")
            st.pyplot(fig, use_container_width=True)

    with c4:
        st.write("Residual ratio distribution where reference new price is available")
        rr = market[
            (market["product_type"] == "Used Car")
            & (market["implied_residual_ratio"].notna())
            & (market["implied_residual_ratio"].between(0, 1.5))
        ]
        if len(rr):
            rr_summary = (
                rr.groupby("brand", as_index=False)
                .agg(median_residual_ratio=("implied_residual_ratio", "median"), observations=("implied_residual_ratio", "size"))
                .query("observations >= 20")
                .sort_values("median_residual_ratio", ascending=False)
                .head(20)
            )
            st.bar_chart(rr_summary.set_index("brand")["median_residual_ratio"])
        else:
            st.info("Residual ratio reference is not available for enough rows.")

with tab3:
    st.subheader("Company Sold Vehicle Benchmarks")
    st.write(
        "This table uses the cleaned sold-vehicle list. It is useful for sanity-checking whether the 2024 Q3 base value is close to observed disposal results."
    )
    sold_view = sold.copy()
    sold_view = sold_view.sort_values("sold_ratio_to_book", ascending=False)
    sold_view["sold_ratio_to_book"] = sold_view["sold_ratio_to_book"].map(lambda x: f"{x*100:.1f}%")
    sold_view["book_car_price"] = sold_view["book_car_price"].map(lambda x: f"${x:,.0f}")
    sold_view["sold_price"] = sold_view["sold_price"].map(lambda x: f"${x:,.0f}")
    st.dataframe(
        sold_view[["Type", "brand", "model_name", "car_year", "segment", "book_car_price", "sold_price", "sold_ratio_to_book", "brand_origin"]].head(100),
        hide_index=True,
        use_container_width=True,
    )

with tab4:
    st.subheader("Methodology")
    st.markdown(
        f"""
        **Data reference period:** 2024 Q3 observed market-price data.  
        **Valuation mode:** 2024 is the observed reference; 2025-2027 are projected values based on depreciation assumptions.  
        **Model type:** Random Forest regression on log-transformed observed car price.

        **Training rows:** {metrics.get("training_rows", 0):,}  
        **Holdout rows:** {metrics.get("test_rows", 0):,}  
        **R² on log price:** {metrics.get("r2_log_price", 0):.3f}  
        **MAE in USD:** ${metrics.get("mae_price_usd", 0):,.0f}  
        **Median absolute error in USD:** ${metrics.get("median_ae_price_usd", 0):,.0f}

        **Input variables**
        - Brand and model
        - New/used product type
        - Car year and vehicle age as of the 2024 Q3 reference period
        - Brand origin
        - Segment/body type
        - Residual value grade, risk grade, and profitability grade where available

        **Projection method**
        - The model first estimates a 2024 Q3 base market value.
        - For 2025, 2026, or 2027, the app applies a data-driven annual depreciation rate to the 2024 Q3 base value.
        - The depreciation rate is estimated from observed median price changes by vehicle age, using a fallback hierarchy: same brand-model, same segment and brand origin, same segment, overall used-car market, then a conservative default.
        - Therefore, a 2026 result should be read as **projected 2026 value based on 2024 Q3 data**, not as a live June 2026 market price.

        **What the app estimates**
        1. 2024 Q3 base market value of the selected car.
        2. Projected market value for the selected valuation year.
        3. Implied residual ratio if a reference new car price is available.
        4. A simple residual risk view using residual ratio and risk grade.

        **Key limitations**
        - Mileage is not included.
        - Accident history is not included.
        - Vehicle condition and service history are not included.
        - Exchange-rate changes, taxes, dealer inventory shocks, import restrictions, and macro market changes after 2024 Q3 are not directly modeled.
        - The model is built from the provided datasets and should not be generalized without further validation.
        - This app excludes customer names, contract numbers, and other unnecessary identifiers.
        """
    )
