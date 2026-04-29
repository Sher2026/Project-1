import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer   # ✅ added

st.set_page_config(page_title="Credit Card Dashboard", layout="wide")

st.title("💳 Credit Card Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("CC GENERAL.csv") 
    return df

df = load_data()

# ================= SIDEBAR =================
menu = st.sidebar.selectbox("Menu", ["Dataset", "Charts", "Manual Entry"])

# ================= DATASET VIEW =================
if menu == "Dataset":
    st.subheader("📊 Full Dataset")
    st.dataframe(df, use_container_width=True)

# ================= CHARTS =================
elif menu == "Charts":
    st.subheader("📈 Data Visualization")

    fig, ax = plt.subplots()
    ax.scatter(df["BALANCE"], df["PURCHASES"])
    ax.set_xlabel("Balance")
    ax.set_ylabel("Purchases")
    st.pyplot(fig)

# ================= MANUAL ENTRY =================
elif menu == "Manual Entry":
    st.subheader("🧾 Enter Customer Data")

    balance = st.number_input("Balance", 0.0)
    purchases = st.number_input("Purchases", 0.0)
    cash_advance = st.number_input("Cash Advance", 0.0)
    credit_limit = st.number_input("Credit Limit", 1.0)

    if st.button("Predict Cluster"):

        # simple features
        input_data = pd.DataFrame([[balance, purchases, cash_advance, credit_limit]],
                                 columns=["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT"])

        # train simple model
        model_data = df[["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT"]]

        # ✅ STEP 1: Handle missing values
        imputer = SimpleImputer(strategy='mean')
        model_data_imputed = imputer.fit_transform(model_data)

        # ✅ STEP 2: Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(model_data_imputed)

        # ✅ STEP 3: Train model
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_scaled)

        # ✅ IMPORTANT: Apply same imputer + scaler to input
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)

        cluster = kmeans.predict(input_scaled)

        st.success(f"🎯 Customer belongs to Cluster: {cluster[0]}")