import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📦 Supply Chain Forecasting & Inventory Optimization")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("forecast_model_final.pkl")
    feature_cols = joblib.load("feature_columns_final.pkl")
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model, feature_columns = load_model()
df = load_data()

# ================= FEATURE ENGINEERING =================
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeakofYear'] = df['Date'].dt.isocalendar().week.astype(int)

df = df.sort_values(["Store ID", "Product ID", "Date"])

df["lag_1"] = df.groupby(["Store ID","Product ID"])["Units Sold"].shift(1)
df["lag_7"] = df.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
df["rolling_mean_7"] = df.groupby(["Store ID","Product ID"])["Units Sold"].transform(
    lambda x: x.rolling(7).mean()
)

df = df.dropna()

# ================= SIDEBAR =================
st.sidebar.header("User Input")

store_id = st.sidebar.selectbox("Select Store", df["Store ID"].unique())

filtered_products = df[df["Store ID"] == store_id]["Product ID"].unique()
product_id = st.sidebar.selectbox("Select Product", filtered_products)

days = st.sidebar.slider("Forecast Days", 7, 60, 30)

data_filtered = df[
    (df["Store ID"] == store_id) &
    (df["Product ID"] == product_id)
].copy()

if data_filtered.empty:
    st.warning("No historical data available for selected Store/Product.")
    st.stop()

if len(data_filtered) < 7:
    st.warning("Minimum 7 historical records required.")
    st.stop()

# ================= FORECAST FUNCTION =================
def forecast_next_days(model, df_current, days):

    df_current = df_current.copy()
    future_list = []
    last_date = df_current["Date"].max()

    for i in range(days):

        next_date = last_date + pd.Timedelta(days=1)

        next_row = df_current[df_current["Date"] == last_date].copy()

        if next_row.empty:
            break

        next_row["Date"] = next_date
        next_row["Year"] = next_date.year
        next_row["Month"] = next_date.month
        next_row["Day"] = next_date.day
        next_row["WeakofYear"] = int(next_date.isocalendar().week)

        temp = pd.concat([df_current, next_row], ignore_index=True)

        temp["lag_1"] = temp.groupby(["Store ID","Product ID"])["Units Sold"].shift(1)
        temp["lag_7"] = temp.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
        temp["rolling_mean_7"] = temp.groupby(["Store ID","Product ID"])["Units Sold"].transform(
            lambda x: x.rolling(7, closed='left').mean()
        )

        future_row = temp[temp["Date"] == next_date].copy()

        if future_row.empty:
            break

        # --------- KEEP FULL ROW FOR RETURN ---------
        future_row_full = future_row.copy()

        # --------- PREPARE FEATURES FOR MODEL ---------
        future_features = future_row[feature_columns].fillna(0)

        if future_features.shape[0] == 0:
            break

        pred = model.predict(future_features)

        future_row_full["Units Sold"] = pred

        future_list.append(future_row_full)

        df_current = pd.concat([df_current, future_row_full], ignore_index=True)
        last_date = next_date

    if len(future_list) == 0:
        return pd.DataFrame()

    return pd.concat(future_list, ignore_index=True)

# ================= RUN FORECAST =================
if st.button("Generate Forecast"):

    future_data = forecast_next_days(model, data_filtered, days)

    if future_data.empty:
        st.error("Forecast could not be generated.")
        st.stop()

    # ================= PLOT =================
    st.subheader("📈 Demand Forecast")

    fig, ax = plt.subplots()
    ax.plot(future_data["Date"], future_data["Units Sold"])
    ax.set_title("30-Day Demand Forecast")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ================= INVENTORY OPTIMIZATION =================
    st.subheader("📊 Inventory Recommendation")

    avg_demand = future_data["Units Sold"].mean()
    demand_std = future_data["Units Sold"].std()

    LEAD_TIME = 7
    SERVICE_LEVEL = 1.65
    ORDERING_COST = 50
    HOLDING_COST = 2

    safety_stock = SERVICE_LEVEL * demand_std * np.sqrt(LEAD_TIME)
    reorder_point = avg_demand * LEAD_TIME + safety_stock
    annual_demand = avg_demand * 365
    eoq = np.sqrt(2 * annual_demand * ORDERING_COST / HOLDING_COST)

    current_inventory = data_filtered.sort_values("Date").iloc[-1]["Inventory Level"]

    stockout_risk = int(current_inventory < reorder_point)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Daily Demand", f"{avg_demand:.2f}")
        st.metric("Safety Stock", f"{safety_stock:.2f}")
        st.metric("Reorder Point", f"{reorder_point:.2f}")

    with col2:
        st.metric("EOQ", f"{eoq:.2f}")
        st.metric("Current Inventory", f"{current_inventory:.2f}")

    if stockout_risk:
        st.error(f"⚠ Stockout Risk! Recommended Order Quantity: {eoq:.0f}")
    else:
        st.success("✅ Inventory Level is Safe")
