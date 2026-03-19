import requests
import json
import sys
import os
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🧠 Expert Role Prompts
ROLE_PROMPTS = {
    "🧑‍🔬 Data Scientist": "You're a data scientist skilled in operational metrics and visualization.",
    "⚡ Fuel Strategist": "You're a fuel efficiency strategist specializing in RPM, power, and fuel optimization.",
    "🛠️ Maritime Technician": "You're a maritime technician focused on diagnosing performance irregularities.",
    "🚨 Anomaly Specialist": "You're an expert in anomaly detection using ML and SHAP analysis."
}

def get_system_prompt(expert_choice):
    return ROLE_PROMPTS.get(expert_choice, ROLE_PROMPTS["🧑‍🔬 Data Scientist"])

# 🔐 Payload preparation for API send
def prepare_payload(df, use_summary=False):
    full_dict = df.to_dict()
    size_estimate = sys.getsizeof(json.dumps(full_dict))

    if size_estimate > 900000 or use_summary:
        summary_dict = df.describe(include='all').to_dict()
        return summary_dict, True
    return full_dict, False

# 🚀 Send to Groq
def safe_send(df, api_key, expert_choice, use_summary=False):
    data_dict, is_summary = prepare_payload(df, use_summary)
    system_prompt = get_system_prompt(expert_choice)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is my dataset: {json.dumps(data_dict)}. Identify patterns, correlations, anomalies, and efficiency trends."}
    ]

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], is_summary

    except json.JSONDecodeError:
        return f"🔴 Failed to parse JSON. Raw response:\n{response.content.decode(errors='replace')}", is_summary
    except requests.exceptions.RequestException as e:
        return f"🔴 API request error:\n{str(e)}", is_summary
    except KeyError:
        return f"⚠️ Unexpected response structure:\n{response.text}", is_summary

# ⚠️ SHAP-based anomaly detection
def detect_anomalies(df):
    numeric_df = df.select_dtypes(include='number').dropna()
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = iso.fit_predict(numeric_df)

    try:
        explainer = shap.TreeExplainer(iso)  # safer than shap.Explainer
        shap_values = explainer(numeric_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        return fig, df[df['anomaly'] == -1]

    except AttributeError:
        # fallback if explainer is incompatible
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "SHAP explainer not compatible.\nPlease rebuild bundle.",
                ha="center", va="center")
        return fig, df[df['anomaly'] == -1]

# 🧮 Clustering logic
def run_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include="number").dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(numeric_df)
    return df, kmeans

# 🤖 What-If Simulator using unified bundle
def simulate_impact_from_bundle(user_inputs, bundle_path, target):
    """
    Simulates prediction from unified pickle file containing:
    - model
    - scaler
    - explainer
    - feature order
    - performance metrics, etc.
    """
    try:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        model = bundle["model"]
        scaler = bundle["scaler"]
        explainer = bundle.get("explainer")
        feature_order = bundle["features"]

        # Validate input keys
        missing = [feat for feat in feature_order if feat not in user_inputs]
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        # Prepare input
        input_vector = pd.DataFrame([user_inputs], columns=feature_order)
        scaled_input = scaler.transform(input_vector)
        prediction = model.predict(scaled_input)[0]

        input_vector[target] = prediction
        input_vector.index = ["Simulated"]

        try:
            shap_result = explainer(scaled_input) if explainer else None
        except AttributeError:
            shap_result = None  # fallback

    except Exception as e:
        err_df = pd.DataFrame([user_inputs])
        err_df[target] = "Prediction failed"
        return err_df.assign(Error=str(e)), None
