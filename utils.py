import requests
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt

# Expert Role Map
ROLE_PROMPTS = {
    "🧑‍🔬 Data Scientist": "You're a data scientist skilled in operational metrics and visualization.",
    "⚡ Fuel Strategist": "You're a fuel efficiency strategist specializing in RPM, power, and fuel optimization.",
    "🛠️ Maritime Technician": "You're a maritime technician focused on diagnosing performance irregularities.",
    "🚨 Anomaly Specialist": "You're an expert in anomaly detection using ML and SHAP analysis."
}

def get_system_prompt(expert_choice):
    return ROLE_PROMPTS.get(expert_choice, ROLE_PROMPTS["🧑‍🔬 Data Scientist"])

def send_to_grok(data_dict, api_key, expert_choice):
    system_prompt = get_system_prompt(expert_choice)
    url = "https://api.cometapi.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is my dataset: {data_dict}. Identify patterns, correlations, anomalies, and efficiency trends."}
    ]
    response = requests.post(url, headers=headers, json={"model": "grok-3", "messages": messages})
    return response.json()["choices"][0]["message"]["content"]

def detect_anomalies(df):
    numeric_df = df.select_dtypes(include='number').dropna()
    iso = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = iso.fit_predict(numeric_df)

    explainer = shap.Explainer(iso, numeric_df)
    shap_values = explainer(numeric_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    return fig, df[df['anomaly'] == -1]

def run_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include="number").dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(numeric_df)
    return df, kmeans

def simulate_impact(df, metric, value):
    df_sim = df.copy()
    df_sim[metric] = value
    return df_sim.describe()
