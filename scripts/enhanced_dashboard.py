#!/usr/bin/env python3
"""
Enhanced Streamlit Dashboard with Advanced Monitoring and Visualization
Includes real-time alerts, performance metrics, data quality monitoring, and advanced charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json

# Enhanced page configuration
st.set_page_config(
    page_title="Enhanced Supply Chain Risk Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}

.risk-critical { border-left-color: #ff4757; background-color: #ffecec; }
.risk-high { border-left-color: #ffa502; background-color: #fff5e6; }
.risk-medium { border-left-color: #3742fa; background-color: #f0f2ff; }
.risk-low { border-left-color: #2ed573; background-color: #f0fff4; }

.alert-critical {
    background-color: #ff4757;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.alert-high {
    background-color: #ffa502;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.performance-metric {
    background-color: #ddd6fe;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
DASHBOARD_REFRESH_INTERVAL = 30  # seconds

# Session state initialization
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def check_enhanced_api_status():
    """Check enhanced API status with detailed information"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_enhanced_risk_prediction(timeframe_days=30):
    """Get enhanced risk prediction with all features"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "timeframe_days": timeframe_days,
                "include_recommendations": True,
                "confidence_threshold": 0.7,
                "correlation_id": f"dashboard_{int(time.time())}"
            },
            headers={
                "Authorization": "Bearer demo_token_123",
                "Content-Type": "application/json"
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_cache_stats():
    """Get cache statistics from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/cache/stats",
            headers={"Authorization": "Bearer admin_token_456"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_model_status():
    """Get detailed model status"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/status",
            headers={"Authorization": "Bearer demo_token_123"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_risk_gauge(value, title, threshold_colors=None):
    """Create an enhanced risk gauge chart"""
    if threshold_colors is None:
        threshold_colors = [(0, "green"), (0.4, "yellow"), (0.7, "orange"), (0.85, "red")]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.4], 'color': "lightgreen"},
                {'range': [0.4, 0.7], 'color': "yellow"},
                {'range': [0.7, 0.85], 'color': "orange"},
                {'range': [0.85, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.85
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_trend_chart(data_history, title):
    """Create trend chart for historical data"""
    if not data_history:
        return go.Figure()
    
    df = pd.DataFrame(data_history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'] if 'timestamp' in df.columns else range(len(df)),
        y=df['value'] if 'value' in df.columns else df.iloc[:, 0],
        mode='lines+markers',
        name=title,
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def display_performance_metrics(health_data, cache_stats, model_status):
    """Display enhanced performance metrics"""
    st.subheader("🚀 System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uptime_hours = health_data.get('uptime_seconds', 0) / 3600
        st.metric("Uptime", f"{uptime_hours:.1f}h", delta=None)
    
    with col2:
        memory_mb = health_data.get('memory_usage_mb', 0)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB", delta=None)
    
    with col3:
        active_conn = health_data.get('active_connections', 0)
        st.metric("Active Connections", active_conn, delta=None)
    
    with col4:
        if cache_stats:
            hit_ratio = cache_stats.get('hit_ratio', 0) * 100
            st.metric("Cache Hit Ratio", f"{hit_ratio:.1f}%", delta=None)
    
    # Cache statistics
    if cache_stats:
        st.markdown("### 💾 Cache Performance")
        cache_col1, cache_col2, cache_col3 = st.columns(3)
        
        with cache_col1:
            st.markdown("**Memory Cache**")
            memory_cache = cache_stats.get('memory_cache', {})
            st.write(f"Size: {memory_cache.get('size', 0)}/{memory_cache.get('maxsize', 0)}")
            st.write(f"TTL: {memory_cache.get('ttl', 0)}s")
        
        with cache_col2:
            st.markdown("**Cache Hits/Misses**")
            hits = cache_stats.get('cache_hits', 0)
            misses = cache_stats.get('cache_misses', 0)
            st.write(f"Hits: {hits}")
            st.write(f"Misses: {misses}")
        
        with cache_col3:
            st.markdown("**Redis Status**")
            redis_info = cache_stats.get('redis', {})
            status = "✅ Connected" if redis_info.get('connected') else "❌ Disconnected"
            st.write(status)
    
    # Model performance
    if model_status and 'models' in model_status:
        st.markdown("### 🤖 Model Status")
        model_df = pd.DataFrame.from_dict(model_status['models'], orient='index')
        if not model_df.empty:
            st.dataframe(model_df, use_container_width=True)

def display_alerts_and_notifications():
    """Display alerts and notifications"""
    st.subheader("🚨 Alerts & Notifications")
    
    # Simulated alerts for demo
    current_alerts = [
        {"severity": "HIGH", "message": "Supply chain disruption detected in Asia-Pacific region", "time": "2 min ago"},
        {"severity": "MEDIUM", "message": "Weather conditions affecting logistics in Europe", "time": "15 min ago"},
        {"severity": "LOW", "message": "Routine supplier performance review completed", "time": "1 hour ago"}
    ]
    
    for alert in current_alerts:
        severity = alert['severity']
        if severity == "CRITICAL":
            st.markdown(f'<div class="alert-critical">🔴 {alert["message"]} - {alert["time"]}</div>', unsafe_allow_html=True)
        elif severity == "HIGH":
            st.markdown(f'<div class="alert-high">🟡 {alert["message"]} - {alert["time"]}</div>', unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ {alert['message']} - {alert['time']}")

def display_data_quality_dashboard():
    """Display data quality monitoring dashboard"""
    st.subheader("📊 Data Quality Monitor")
    
    # Simulated data quality metrics
    quality_data = {
        "Satellite": {"score": 0.94, "issues": ["None"], "last_update": "2 min ago"},
        "Social Media": {"score": 0.87, "issues": ["Rate limiting"], "last_update": "5 min ago"},
        "Weather": {"score": 0.92, "issues": ["None"], "last_update": "1 min ago"},
        "Economic": {"score": 0.89, "issues": ["API latency"], "last_update": "3 min ago"},
        "News": {"score": 0.95, "issues": ["None"], "last_update": "1 min ago"}
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quality scores chart
        sources = list(quality_data.keys())
        scores = [quality_data[source]["score"] for source in sources]
        
        fig = px.bar(
            x=sources, 
            y=scores,
            title="Data Quality Scores by Source",
            color=scores,
            color_continuous_scale="RdYlGn",
            range_color=[0.7, 1.0]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Quality Issues**")
        for source, data in quality_data.items():
            issues = data["issues"]
            if issues == ["None"]:
                st.success(f"✅ {source}: No issues")
            else:
                st.warning(f"⚠️ {source}: {', '.join(issues)}")

def create_correlation_matrix():
    """Create correlation matrix for risk factors"""
    # Simulated correlation data
    factors = ['Satellite', 'Social', 'Weather', 'Economic', 'News']
    correlation_data = np.random.rand(5, 5)
    correlation_data = (correlation_data + correlation_data.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_data, 1)  # Diagonal should be 1
    
    fig = px.imshow(
        correlation_data,
        labels=dict(x="Risk Factors", y="Risk Factors", color="Correlation"),
        x=factors,
        y=factors,
        title="Risk Factor Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    
    return fig

# Main dashboard
def main():
    st.title("🔗 Enhanced Supply Chain Risk Intelligence Dashboard")
    st.markdown("*Real-time monitoring with advanced analytics and performance optimization*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 120, 30)
    
    # Prediction settings
    timeframe = st.sidebar.slider("📅 Prediction Timeframe (days)", 1, 90, 30)
    confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.5, 0.95, 0.7)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        show_uncertainty = st.checkbox("Show uncertainty metrics", True)
        show_model_contributions = st.checkbox("Show model contributions", True)
        enable_alerts = st.checkbox("Enable real-time alerts", True)
    
    # Check system status
    health_data = check_enhanced_api_status()
    api_status = health_data is not None
    
    # Status indicator
    if api_status:
        st.success("✅ System Online - All services operational")
    else:
        st.error("❌ System Offline - Please check API server")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Risk Overview", "📈 Analytics", "🔧 Performance", "📊 Data Quality", "🚨 Alerts"])
    
    with tab1:
        # Risk prediction and gauges
        prediction_data = get_enhanced_risk_prediction(timeframe)
        
        if prediction_data:
            risk_scores = prediction_data['risk_scores']
            
            # Main risk gauges
            col1, col2, col3 = st.columns(3)
            
            with col1:
                overall_risk = risk_scores.get('overall_risk', 0.5)
                fig1 = create_risk_gauge(overall_risk, "Overall Risk")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                operational_risk = risk_scores.get('operational_risk', 0.4)
                fig2 = create_risk_gauge(operational_risk, "Operational Risk")
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                financial_risk = risk_scores.get('financial_risk', 0.3)
                fig3 = create_risk_gauge(financial_risk, "Financial Risk")
                st.plotly_chart(fig3, use_container_width=True)
            
            # Individual model predictions
            if 'individual_predictions' in prediction_data:
                st.subheader("🤖 Individual Model Predictions")
                
                individual_preds = prediction_data['individual_predictions']
                pred_df = pd.DataFrame.from_dict(individual_preds, orient='index', columns=['Prediction'])
                
                fig = px.bar(
                    x=pred_df.index,
                    y=pred_df['Prediction'],
                    title="Model-wise Risk Predictions",
                    color=pred_df['Prediction'],
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if 'recommendations' in prediction_data:
                st.subheader("💡 Smart Recommendations")
                for i, rec in enumerate(prediction_data['recommendations']):
                    if "CRITICAL" in rec:
                        st.error(f"🚨 {rec}")
                    elif "HIGH" in rec:
                        st.warning(f"⚠️ {rec}")
                    elif "MEDIUM" in rec:
                        st.info(f"ℹ️ {rec}")
                    else:
                        st.success(f"✅ {rec}")
        
    with tab2:
        st.subheader("📈 Advanced Analytics")
        
        # Correlation matrix
        corr_fig = create_correlation_matrix()
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Time series trends (simulated)
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk trend over time
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            risk_trend = np.random.uniform(0.3, 0.8, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=risk_trend,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(title="Risk Trend (30 Days)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence trend
            confidence_trend = np.random.uniform(0.7, 0.95, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_trend,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(title="Prediction Confidence (30 Days)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Performance metrics
        cache_stats = get_cache_stats()
        model_status = get_model_status()
        display_performance_metrics(health_data, cache_stats, model_status)
        
        # Performance trends
        if prediction_data and 'prediction_metrics' in prediction_data:
            metrics = prediction_data['prediction_metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{metrics.get('processing_time_ms', 0):.1f} ms")
            with col2:
                st.metric("Model Agreement", f"{metrics.get('model_agreement', 0):.3f}")
            with col3:
                st.metric("Prediction Uncertainty", f"{metrics.get('prediction_uncertainty', 0):.3f}")
    
    with tab4:
        display_data_quality_dashboard()
    
    with tab5:
        display_alerts_and_notifications()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Enhanced Supply Chain Risk Intelligence Dashboard v2.0** | *Powered by Multi-modal AI*")

if __name__ == "__main__":
    main()
