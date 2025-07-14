import streamlit as st
import pandas as pd
import os
from utils import send_to_grok, detect_anomalies, run_clustering, simulate_impact

st.set_page_config(page_title="Grok Operational Analyzer", layout="wide")

st.title("📊 Grok Agentic Operational Analyzer")

# 🔑 Load API Key
grok_key = st.secrets.get("GROK_API_KEY", os.getenv("GROK_API_KEY"))

# 🧠 Expert Role Selector
role_options = {
    "🧑‍🔬 Data Scientist": "Skilled in pattern discovery, correlations, and visualization.",
    "⚡ Fuel Strategist": "Optimizes fuel consumption and operational regimes.",
    "🛠️ Maritime Technician": "Diagnoses engine performance irregularities.",
    "🚨 Anomaly Specialist": "Detects outliers and explains them using ML and SHAP."
}
expert_choice = st.sidebar.selectbox("👨‍🏫 Select Expert", list(role_options.keys()))
st.sidebar.info(f"ℹ️ {role_options[expert_choice]}")

# 📁 Data Upload
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Upload", "🧪 Sample Mode", "📊 Full Dataset", "⚠️ Anomaly Detection",
    "🧠 Grok Summary", "🧮 Clustering", "🤔 What-If Simulator"
])

with tab1:
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ File uploaded successfully.")
        st.dataframe(df.head())

with tab2:
    if "df" in st.session_state:
        sample = st.session_state["df"].head(20)
        st.subheader("🧪 Sample Mode")
        st.dataframe(sample)
        if st.button("Analyze Sample with Grok"):
            response = send_to_grok(sample.to_dict(), grok_key, expert_choice)
            st.markdown("### 🔍 Grok Insights")
            st.json(response)

with tab3:
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("📊 Full Dataset Mode")
        st.line_chart(df.select_dtypes('number'))
        if st.button("Analyze Full Dataset with Grok"):
            response = send_to_grok(df.to_dict(), grok_key, expert_choice)
            st.markdown("### 🧠 Grok Full Analysis")
            st.json(response)

with tab4:
    if "df" in st.session_state:
        shap_fig, anomalies = detect_anomalies(st.session_state["df"])
        st.subheader("⚠️ SHAP Anomaly Detection")
        st.pyplot(shap_fig)
        st.markdown("### 🚨 Detected Anomalies")
        st.dataframe(anomalies)

with tab5:
    if "df" in st.session_state:
        st.subheader("🧠 Narrative Summary")
        response = send_to_grok(st.session_state["df"].to_dict(), grok_key, expert_choice)
        st.write(response)

with tab6:
    if "df" in st.session_state:
        st.subheader("🧮 KMeans Clustering")
        df = st.session_state["df"]
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        clustered_df, model = run_clustering(df, n_clusters)
        st.dataframe(clustered_df)
        selected_feature = st.selectbox("Choose feature for x-axis", df.select_dtypes("number").columns)
        st.scatter_chart(clustered_df, x=selected_feature, y="FOC", color="cluster")

with tab7:
    if "df" in st.session_state:
        st.subheader("🤔 What-If Simulation")
        df = st.session_state["df"]
        metric = st.selectbox("Select Metric", df.select_dtypes("number").columns)
        value = st.number_input(f"Set {metric} value for simulation", value=float(df[metric].mean()))
        if st.button("Run Simulation"):
            result = simulate_impact(df, metric, value)
            st.write(f"📊 Impact Summary for {metric} = {value}")
            st.dataframe(result)
