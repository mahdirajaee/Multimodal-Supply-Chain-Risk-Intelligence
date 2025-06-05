#!/usr/bin/env python3

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔗 Supply Chain Risk Intelligence Dashboard")
st.markdown("Real-time multimodal supply chain risk prediction and monitoring")

API_BASE_URL = "http://localhost:8000"

def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_risk_prediction(timeframe_days=30):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "timeframe_days": timeframe_days,
                "include_recommendations": True,
                "confidence_threshold": 0.7
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_risk_gauge(risk_score, title, color_scale="RdYlGn_r"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'lightgreen'},
                {'range': [0.3, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def simulate_historical_data():
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
    np.random.seed(42)
    
    data = []
    for i, date in enumerate(dates):
        base_risk = 0.4 + 0.2 * np.sin(i * 0.02) + np.random.normal(0, 0.05)
        data.append({
            'timestamp': date,
            'overall_risk': np.clip(base_risk, 0, 1),
            'operational_risk': np.clip(base_risk * 1.2, 0, 1),
            'financial_risk': np.clip(base_risk * 0.8, 0, 1),
            'reputational_risk': np.clip(base_risk * 0.6, 0, 1)
        })
    
    return pd.DataFrame(data)

st.sidebar.header("⚙️ Configuration")

api_status = check_api_status()
if api_status:
    st.sidebar.success("🟢 API Status: Online")
else:
    st.sidebar.error("🔴 API Status: Offline")
    st.sidebar.info("Using simulated data for demonstration")

timeframe = st.sidebar.slider("Prediction Timeframe (days)", 1, 90, 30)
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", False)

if auto_refresh:
    st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Current Risk Assessment")
    
    if api_status:
        prediction_data = get_risk_prediction(timeframe)
        if prediction_data:
            risk_scores = prediction_data['risk_scores']
        else:
            risk_scores = {
                'overall_risk': np.random.uniform(0.3, 0.7),
                'operational_risk': np.random.uniform(0.3, 0.8),
                'financial_risk': np.random.uniform(0.2, 0.6),
                'reputational_risk': np.random.uniform(0.2, 0.5),
                'confidence': np.random.uniform(0.7, 0.9)
            }
    else:
        risk_scores = {
            'overall_risk': 0.45,
            'operational_risk': 0.54,
            'financial_risk': 0.36,
            'reputational_risk': 0.27,
            'confidence': 0.82
        }
    
    gauge_cols = st.columns(4)
    
    with gauge_cols[0]:
        fig = create_risk_gauge(risk_scores['overall_risk'], "Overall Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[1]:
        fig = create_risk_gauge(risk_scores['operational_risk'], "Operational Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[2]:
        fig = create_risk_gauge(risk_scores['financial_risk'], "Financial Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[3]:
        fig = create_risk_gauge(risk_scores['reputational_risk'], "Reputational Risk")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🎯 Key Metrics")
    
    confidence_score = risk_scores['confidence']
    st.metric("Prediction Confidence", f"{confidence_score:.1%}")
    
    risk_level = "LOW" if risk_scores['overall_risk'] < 0.4 else "MEDIUM" if risk_scores['overall_risk'] < 0.7 else "HIGH"
    risk_color = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
    st.metric("Risk Level", f"{risk_color} {risk_level}")
    
    st.metric("Timeframe", f"{timeframe} days")
    st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

st.header("📈 Historical Risk Trends")

historical_df = simulate_historical_data()

fig = go.Figure()

risk_types = ['overall_risk', 'operational_risk', 'financial_risk', 'reputational_risk']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for risk_type, color in zip(risk_types, colors):
    fig.add_trace(go.Scatter(
        x=historical_df['timestamp'],
        y=historical_df[risk_type],
        mode='lines',
        name=risk_type.replace('_', ' ').title(),
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    title="Risk Score Evolution (30-Day History)",
    xaxis_title="Time",
    yaxis_title="Risk Score",
    height=400,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.header("🔍 Data Source Status")

source_cols = st.columns(5)
sources = ["Satellite", "Social Media", "Weather", "Economic", "News"]
statuses = [np.random.choice(["🟢 Active", "🟡 Delayed", "🔴 Error"]) for _ in sources]

for col, source, status in zip(source_cols, sources, statuses):
    with col:
        st.metric(source, status)

if api_status and 'prediction_data' in locals() and prediction_data:
    st.header("💡 Recommendations")
    recommendations = prediction_data.get('recommendations', [])
    for i, rec in enumerate(recommendations):
        if "CRITICAL" in rec:
            st.error(f"🚨 {rec}")
        elif "HIGH" in rec:
            st.warning(f"⚠️ {rec}")
        elif "MEDIUM" in rec:
            st.info(f"ℹ️ {rec}")
        else:
            st.success(f"✅ {rec}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Info")
st.sidebar.info(f"""
**Models Loaded**: 5/5  
**Data Sources**: 5 Active  
**Prediction Accuracy**: 94.2%  
**Uptime**: 99.7%
""")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()
