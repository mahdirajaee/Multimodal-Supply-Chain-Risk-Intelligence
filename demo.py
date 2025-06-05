#!/usr/bin/env python3
"""
Multimodal Supply Chain Risk Intelligence System Demo
Showcases the complete system with real-time predictions
"""

import requests
import json
import time
from datetime import datetime

# API base URL
API_BASE = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    print(f"\n--- {title} ---")

def test_system_health():
    print_header("SYSTEM HEALTH CHECK")
    
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ System Status: {data['status']}")
        print(f"✅ Timestamp: {data['timestamp']}")
    else:
        print("❌ Health check failed")
        return False
    
    response = requests.get(f"{API_BASE}/models/status")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model Status: {data['status']}")
        print(f"✅ Loaded Models: {', '.join(data['models'])}")
        print(f"✅ Last Updated: {data['last_updated']}")
    else:
        print("❌ Model status check failed")
        return False
    
    return True

def demo_risk_prediction():
    print_header("RISK PREDICTION DEMO")
    
    # Test different scenarios
    scenarios = [
        {"timeframe_days": 7, "scenario": "Short-term (1 week)"},
        {"timeframe_days": 30, "scenario": "Medium-term (1 month)"},
        {"timeframe_days": 90, "scenario": "Long-term (3 months)"}
    ]
    
    for scenario in scenarios:
        print_section(f"{scenario['scenario']} Risk Assessment")
        
        response = requests.post(
            f"{API_BASE}/predict",
            json={"timeframe_days": scenario["timeframe_days"]},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📊 Overall Risk Score: {data['risk_scores']['overall_risk']:.3f}")
            print(f"🏭 Operational Risk: {data['risk_scores']['operational_risk']:.3f}")
            print(f"💰 Financial Risk: {data['risk_scores']['financial_risk']:.3f}")
            print(f"📈 Reputational Risk: {data['risk_scores']['reputational_risk']:.3f}")
            print(f"🎯 Confidence: {data['risk_scores']['confidence']:.3f}")
            
            print("\n🤖 Individual Model Predictions:")
            for model, score in data['individual_predictions'].items():
                print(f"  • {model.capitalize()}: {score:.3f}")
            
            print("\n💡 Recommendations:")
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"❌ Prediction failed: {response.text}")

def demo_real_time_monitoring():
    print_header("REAL-TIME MONITORING SIMULATION")
    
    print("🔄 Simulating real-time supply chain monitoring...")
    print("   (Making predictions every 10 seconds for 30 seconds)")
    
    for i in range(3):
        print(f"\n⏰ Update #{i+1} at {datetime.now().strftime('%H:%M:%S')}")
        
        response = requests.post(
            f"{API_BASE}/predict",
            json={"timeframe_days": 30},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            overall_risk = data['risk_scores']['overall_risk']
            confidence = data['risk_scores']['confidence']
            
            # Risk level interpretation
            if overall_risk >= 0.8:
                level = "🔴 CRITICAL"
            elif overall_risk >= 0.6:
                level = "🟡 HIGH"
            elif overall_risk >= 0.4:
                level = "🟠 MEDIUM"
            else:
                level = "🟢 LOW"
            
            print(f"   Risk Level: {level} ({overall_risk:.3f})")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Primary Recommendation: {data['recommendations'][0]}")
        
        if i < 2:  # Don't sleep after the last iteration
            time.sleep(10)

def display_system_summary():
    print_header("SYSTEM CAPABILITIES SUMMARY")
    
    print("🎯 MULTIMODAL AI MODELS:")
    print("  • Satellite Imagery Analysis (Vision Transformer)")
    print("  • Social Media Sentiment (BERT)")
    print("  • Weather Pattern Analysis (LSTM)")
    print("  • Economic Indicator Analysis (Transformer)")
    print("  • News Event Classification (RoBERTa)")
    
    print("\n🔧 FUSION ENGINE:")
    print("  • Multi-head attention mechanism")
    print("  • Dynamic feature projection")
    print("  • Confidence scoring")
    print("  • Risk decomposition")
    
    print("\n📊 API ENDPOINTS:")
    print("  • GET /health - System health check")
    print("  • GET /models/status - Model status")
    print("  • POST /predict - Risk prediction")
    
    print("\n📈 DASHBOARD FEATURES:")
    print("  • Real-time risk gauges")
    print("  • Historical trend analysis")
    print("  • Interactive visualizations")
    print("  • Model performance metrics")
    
    print("\n🌐 ACCESS POINTS:")
    print("  • API Server: http://localhost:8000")
    print("  • Interactive Dashboard: http://localhost:8502")

def main():
    print_header("MULTIMODAL SUPPLY CHAIN RISK INTELLIGENCE SYSTEM")
    print("🚀 Advanced AI-powered supply chain risk monitoring and prediction")
    
    # Test system health first
    if not test_system_health():
        print("\n❌ System health check failed. Please ensure the API server is running.")
        print("   Run: cd /path/to/project && python main.py mode=api")
        return
    
    # Demo core functionality
    demo_risk_prediction()
    
    # Real-time monitoring simulation
    demo_real_time_monitoring()
    
    # System summary
    display_system_summary()
    
    print_header("DEMO COMPLETE")
    print("✅ The Multimodal Supply Chain Risk Intelligence System is fully operational!")
    print("🌐 Access the interactive dashboard at: http://localhost:8502")
    print("📚 API documentation available at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
