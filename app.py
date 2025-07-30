import streamlit as st
import pandas as pd
import os
from utils import safe_send, detect_anomalies, run_clustering, simulate_impact

st.set_page_config(page_title="Grok Agentic Operational Analyzer", layout="wide")
st.title("📊 Grok Agentic Operational Analyzer")

# 🔑 Load Groq API Key securely
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# 🧠 Expert Role Selector
role_options = {
    "🧑‍🔬 Data Scientist": "Skilled in pattern discovery, correlations, and visualization.",
    "⚡ Fuel Strategist": "Optimizes fuel consumption and operational regimes.",
    "🛠️ Maritime Technician": "Diagnoses engine performance irregularities.",
    "🚨 Anomaly Specialist": "Detects outliers and explains them using ML and SHAP."
}
expert_choice = st.sidebar.selectbox("👨‍🏫 Select Expert", list(role_options.keys()))
st.sidebar.info(f"ℹ️ {role_options[expert_choice]}")

# 📁 Data Upload and Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Upload", "🧪 Sample Mode", "📊 Full Dataset", "⚠️ Anomaly Detection",
    "🧠 Groq Summary", "🧮 Clustering", "🤔 What-If Simulator"
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
        if st.button("Analyze Sample with Groq"):
            response, is_summary = safe_send(sample, groq_key, expert_choice, use_summary=False)
            if is_summary:
                st.warning("⚠️ Summary sent due to size or toggle.")
            st.markdown("### 🔍 Groq Insights")
            st.write(response)

with tab3:
    st.subheader("📊 Groq Full Dataset Analysis")

    use_summary = st.toggle("Use summary for full dataset?", value=True)
    
    if st.button("Analyze Full Dataset with Groq"):
        response, is_summary = safe_send(df, groq_key, expert_choice, use_summary=use_summary)
        if is_summary:
            st.warning("⚠️ Summary sent due to size or toggle.")
        st.markdown("### 🧠 Groq Full Analysis")
        st.write(response)


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
        df = st.session_state["df"]
        response, is_summary = safe_send(df, groq_key, expert_choice, use_summary=True)
        if is_summary:
            st.info("ℹ️ Using statistical summary for this analysis.")
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
        numeric_cols = df.select_dtypes("number").columns.tolist()
        target_col = st.selectbox("🎯 Target Metric to Predict", numeric_cols, index=numeric_cols.index("FOC") if "FOC" in numeric_cols else 0)

        st.markdown("⚙️ Modify the following inputs:")
        user_inputs = {}
        for col in numeric_cols:
            if col != target_col:
                val = st.number_input(f"{col}", value=float(df[col].mean()))
                user_inputs[col] = val

        if st.button("Run What-If Simulation"):
            result = simulate_impact(df, user_inputs, target=target_col)
            st.success(f"✅ Predicted {target_col} = {result[target_col].values[0]:.2f}")
            st.subheader("📊 Simulated Operational Metrics")
            st.dataframe(result)
