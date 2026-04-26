import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from predict import predict, load_models
from utils import get_risk_level, COLUMN_NAMES

# --- Page Config ---
st.set_page_config(
    page_title="Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .stApp {
        background: #0a0e1a;
        color: #e0e6f0;
    }
    .main-header {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7a99;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1e2d40;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-HIGH {
        background: #2d0a0a;
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #ff6b6b;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .risk-MEDIUM {
        background: #1f1a0a;
        border: 1px solid #ffaa00;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #ffcc44;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .risk-LOW {
        background: #0a1f0a;
        border: 1px solid #00cc44;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #44ff88;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .alert-box {
        background: #2d0a0a;
        border-left: 4px solid #ff4444;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #ffaaaa;
    }
    div[data-testid="stSidebar"] {
        background: #080c18 !important;
        border-right: 1px solid #1e2d40;
    }
    .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    try:
        return load_models()
    except Exception as e:
        return None


def plot_attack_distribution(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#111827')
    
    palette = {'normal': '#00cc44', 'dos': '#ff4444', 'probe': '#ffaa00',
               'r2l': '#7b61ff', 'u2r': '#00d4ff', 'unknown': '#888888'}

    # Attack category distribution
    cat_counts = results_df['predicted_category'].value_counts()
    colors = [palette.get(c, '#888888') for c in cat_counts.index]
    axes[0].bar(cat_counts.index, cat_counts.values, color=colors, edgecolor='#1e2d40')
    axes[0].set_title('Attack Category Distribution', color='white', fontsize=13)
    axes[0].set_facecolor('#111827')
    axes[0].tick_params(colors='#aaaaaa')
    for spine in axes[0].spines.values():
        spine.set_color('#1e2d40')

    # Risk level pie
    risk_counts = results_df['risk_level'].value_counts()
    risk_colors = {'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00cc44'}
    pie_colors = [risk_colors.get(r, '#888') for r in risk_counts.index]
    axes[1].pie(risk_counts.values, labels=risk_counts.index, colors=pie_colors,
                autopct='%1.1f%%', textprops={'color': 'white'}, startangle=90)
    axes[1].set_title('Risk Level Distribution', color='white', fontsize=13)
    axes[1].set_facecolor('#111827')

    plt.tight_layout()
    return fig


def plot_anomaly_scores(results_df):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    
    colors = ['#ff4444' if a == 1 else '#00cc44' for a in results_df['is_anomaly']]
    ax.scatter(range(len(results_df)), results_df['anomaly_score'], c=colors, alpha=0.6, s=20)
    ax.axhline(0, color='#ffaa00', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_title('Anomaly Scores per Sample', color='white', fontsize=13)
    ax.tick_params(colors='#aaaaaa')
    ax.set_xlabel('Sample Index', color='#aaaaaa')
    ax.set_ylabel('Anomaly Score', color='#aaaaaa')
    ax.legend(facecolor='#111827', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#1e2d40')
    
    plt.tight_layout()
    return fig


def show_performance_metrics():
    plots_path = 'plots'
    tabs = st.tabs(["Random Forest CM", "Logistic Regression CM", "Feature Importance"])
    
    with tabs[0]:
        fp = os.path.join(plots_path, 'rf_confusion.png')
        if os.path.exists(fp):
            st.image(fp, caption="Random Forest Confusion Matrix", use_column_width=True)
        else:
            st.info("Run train.py first to generate plots.")
    
    with tabs[1]:
        fp = os.path.join(plots_path, 'lr_confusion.png')
        if os.path.exists(fp):
            st.image(fp, caption="Logistic Regression Confusion Matrix", use_column_width=True)
        else:
            st.info("Run train.py first to generate plots.")
    
    with tabs[2]:
        fp = os.path.join(plots_path, 'feature_importance.png')
        if os.path.exists(fp):
            st.image(fp, caption="Top Feature Importances", use_column_width=True)
        else:
            st.info("Run train.py first to generate plots.")


# ======================== MAIN APP ========================

st.markdown('<div class="main-header">🛡️ Cyber Threat Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NSL-KDD · Classical ML · Random Forest + Isolation Forest</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox("Classification Model", ["random_forest", "logistic_regression"],
                                 format_func=lambda x: "Random Forest" if x == "random_forest" else "Logistic Regression")
    st.markdown("---")
    st.markdown("### 📖 Risk Levels")
    st.markdown("🔴 **High** — prob > 0.7")
    st.markdown("🟡 **Medium** — 0.3–0.7")
    st.markdown("🟢 **Low** — prob < 0.3")
    st.markdown("---")
    st.markdown("### 📁 Expected CSV Format")
    st.markdown("Upload NSL-KDD format CSV with 41 feature columns.")

# Check models loaded
artifacts = get_models()
if artifacts is None:
    st.error("⚠️ Models not found. Please run `python train.py` first.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 Model Performance", "ℹ️ Sample Format"])

# ---- TAB 1: Prediction ----
with tab1:
    uploaded = st.file_uploader("Upload Network Traffic CSV", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        
        # Handle if file has header
        if df.iloc[0, 0] == 'duration' or df.columns[0] == 'duration':
            df = pd.read_csv(uploaded)
        else:
            cols = COLUMN_NAMES[:df.shape[1]]
            df.columns = cols
        
        st.markdown(f"**Loaded {len(df)} records**")
        st.dataframe(df.head(5), use_container_width=True)
        
        if st.button("🚀 Run Prediction", type="primary"):
            with st.spinner("Analyzing traffic..."):
                try:
                    results = predict(df, model_choice=model_choice)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total = len(results)
                    attacks = results['is_attack'].sum()
                    high_risk = (results['risk_level'] == 'High').sum()
                    anomalies = results['is_anomaly'].sum()
                    
                    col1.metric("Total Records", total)
                    col2.metric("Attacks Detected", attacks, f"{attacks/total:.1%}")
                    col3.metric("High Risk", high_risk, delta_color="inverse")
                    col4.metric("Anomalies", anomalies)
                    
                    st.markdown("---")
                    
                    # Charts
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### Attack & Risk Distribution")
                        st.pyplot(plot_attack_distribution(results))
                    with col_b:
                        st.markdown("#### Anomaly Scores")
                        st.pyplot(plot_anomaly_scores(results))
                    
                    st.markdown("---")
                    
                    # Alert section
                    high_risk_df = results[results['risk_level'] == 'High']
                    if len(high_risk_df) > 0:
                        st.markdown("### 🚨 High Risk Alerts")
                        for idx, row in high_risk_df.head(10).iterrows():
                            st.markdown(
                                f'<div class="alert-box">⚠️ Record #{idx} | '
                                f'Category: <b>{row["predicted_category"].upper()}</b> | '
                                f'Attack Prob: {row["attack_probability"]:.1%} | '
                                f'Anomaly: {"YES" if row["is_anomaly"] else "NO"}</div>',
                                unsafe_allow_html=True
                            )
                        if len(high_risk_df) > 10:
                            st.caption(f"...and {len(high_risk_df) - 10} more high-risk records")
                    else:
                        st.success("✅ No high-risk threats detected.")
                    
                    st.markdown("---")
                    st.markdown("#### Full Prediction Results")
                    
                    # Color-code risk in display
                    st.dataframe(results, use_container_width=True)
                    
                    # Download
                    csv_out = results.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results CSV", csv_out, "predictions.csv", "text/csv")
                
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)
    else:
        st.info("👆 Upload a CSV file to begin threat analysis.")

# ---- TAB 2: Performance ----
with tab2:
    st.markdown("### Model Performance Metrics")
    st.markdown("These plots are generated during training (`python train.py`).")
    show_performance_metrics()

# ---- TAB 3: Sample Format ----
with tab3:
    st.markdown("### Sample Input Format")
    st.markdown("""
    The system expects NSL-KDD format data. Here are the columns:
    """)
    
    from predict import SAMPLE_NORMAL, SAMPLE_ATTACK
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sample Normal Traffic**")
        st.json(SAMPLE_NORMAL)
    with col2:
        st.markdown("**Sample Attack Traffic (DoS/Smurf)**")
        st.json(SAMPLE_ATTACK)
    
    st.markdown("---")
    st.markdown("**Download sample CSVs to test:**")
    
    sample_df = pd.DataFrame([SAMPLE_NORMAL, SAMPLE_ATTACK])
    st.download_button(
        "⬇️ Download Sample CSV",
        sample_df.to_csv(index=False, header=False).encode(),
        "sample_traffic.csv",
        "text/csv"
    )
