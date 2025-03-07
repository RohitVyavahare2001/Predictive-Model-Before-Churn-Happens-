import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        "customer_id": ["CUST-00001", "CUST-00002", "CUST-00003", "CUST-00004", "CUST-00005"],
        "tenure": [12, 24, 36, 48, 60],
        "monthly_charges": [50, 70, 90, 110, 130],
        "total_charges": [600, 1680, 3240, 5280, 7800],
        "avg_session_length": [30, 40, 35, 50, 60],
        "avg_session_count_monthly": [10, 15, 12, 18, 20],
        "days_since_last_activity": [5, 10, 7, 15, 20],
        "support_tickets_count": [1, 2, 1, 3, 4],
        "items_purchased_6m": [2, 5, 3, 6, 7],
        "avg_purchase_value": [100, 200, 150, 300, 400],
        "contract_type": ["Monthly", "One-year", "Two-year", "Monthly", "One-year"],
        "payment_method": ["Credit card", "Bank transfer", "Electronic check", "Mailed check", "Credit card"],
        "subscription_tier": ["Basic", "Standard", "Premium", "Standard", "Premium"],
        "has_premium_support": [False, True, False, True, True],
        "has_family_plan": [True, False, True, False, True],
        "has_multiple_devices": [True, True, False, False, True]
    })

def get_predictions(data):
    response = requests.post(API_URL, json={"customers": data.to_dict(orient="records"), "risk_threshold": 0.5})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

data = load_sample_data()

data_placeholder = st.empty()
if st.button("Predict Churn"):
    with st.spinner("Fetching predictions..."):
        predictions = get_predictions(data)
        if predictions:
            data["Churn Probability"] = [p["churn_probability"] for p in predictions]
            data["High Risk"] = [p["is_high_risk"] for p in predictions]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(data))
            col2.metric("High-Risk Customers", sum(data["High Risk"]))
            col3.metric("MRR at Risk ($)", round(sum(data[data["High Risk"]]["monthly_charges"])))
            
            data_placeholder.dataframe(data, use_container_width=True)

if "Churn Probability" in data:
    fig = px.histogram(data, x="Churn Probability", nbins=10, title="Churn Probability Distribution", color_discrete_sequence=["#FF5733"])
    st.plotly_chart(fig, use_container_width=True)

    risk_counts = data["High Risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig_risk = px.pie(risk_counts, names="Risk Level", values="Count", title="Churn Risk Segmentation", color_discrete_sequence=["#FF5733", "#33FF57"])
    st.plotly_chart(fig_risk, use_container_width=True)

    fig_factors = px.bar(data.melt(id_vars=["customer_id"], value_vars=["tenure", "monthly_charges", "support_tickets_count"], var_name="Feature", value_name="Value"), x="Feature", y="Value", title="Top Churn Factors", color="Feature")
    st.plotly_chart(fig_factors, use_container_width=True)
