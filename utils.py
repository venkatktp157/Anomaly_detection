import requests
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt
import json
import sys
import pandas as pd

# 🧠 Expert Role Prompts
ROLE_PROMPTS = {
    "🧑‍🔬 Data Scientist": "You're a data scientist skilled in operational metrics and visualization.",
    "⚡ Fuel Strategist": "You're a fuel efficiency strategist specializing in RPM, power, and fuel optimization.",
    "🛠️ Maritime Technician": "You're a maritime technician focused on diagnosing performance irregularities.",
    "🚨 Anomaly Specialist": "You're an expert in anomaly detection using ML and SHAP analysis."
}

def get_system_prompt(expert_choice):
    return ROLE_PROMPTS.get(expert_choice, ROLE_PROMPTS["🧑‍🔬 Data Scientist"])

# 🧩 NEW: Payload Preparer
def prepare_payload(df, use_summary=False):
    """
    Determines whether to send full or summarized data based on size or toggle.
    """
    full_dict = df.to_dict()
    size_estimate = sys.getsizeof(json.dumps(full_dict))

    if size_estimate > 900000 or use_summary:
        summary_dict = df.describe(include='all').to_dict()
        return summary_dict, True  # True = summary mode
    return full_dict, False  # False = full data mode

# 🚀 NEW: Modularized safe_send with Groq
def safe_send(df, api_key, expert_choice, use_summary=False):
    """
    Sends data to Groq API, gracefully falling back to summary mode.
    """
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
        "model": "llama3-8b-8192",
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

# ⚠️ SHAP Anomaly Detection
def detect_anomalies(df):
    numeric_df = df.select_dtypes(include='number').dropna()
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = iso.fit_predict(numeric_df)

    explainer = shap.Explainer(iso, numeric_df)
    shap_values = explainer(numeric_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    return fig, df[df['anomaly'] == -1]

# 🧮 Clustering
def run_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include="number").dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(numeric_df)
    return df, kmeans

# 🤔 Impact Simulator
from sklearn.linear_model import LinearRegression

def simulate_impact(df, feature_values, target="FOC"):
    df_clean = df.dropna()
    features = df_clean.select_dtypes("number").drop(columns=[target])
    target_values = df_clean[target]

    model = LinearRegression()
    model.fit(features, target_values)

    new_input = features.mean().to_dict()
    new_input.update(feature_values)  # Apply user overrides

    predicted = model.predict([list(new_input.values())])[0]
    new_input[target] = predicted

    return pd.DataFrame([new_input])
