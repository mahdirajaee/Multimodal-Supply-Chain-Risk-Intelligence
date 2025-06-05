#!/usr/bin/env python3
"""
Advanced Alert System for Supply Chain Risk Intelligence
Provides real-time monitoring, notifications, and automated responses
"""

import asyncio
import smtplib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    RISK_THRESHOLD = "risk_threshold"
    CONFIDENCE_DROP = "confidence_drop"
    MODEL_FAILURE = "model_failure"
    TREND_ANOMALY = "trend_anomaly"
    SYSTEM_HEALTH = "system_health"

@dataclass
class Alert:
    id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    risk_score: float
    confidence: float
    affected_models: List[str]
    recommendations: List[str]
    resolved: bool = False
    acknowledged: bool = False

class AlertDatabase:
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    severity TEXT,
                    alert_type TEXT,
                    title TEXT,
                    message TEXT,
                    risk_score REAL,
                    confidence REAL,
                    affected_models TEXT,
                    recommendations TEXT,
                    resolved BOOLEAN,
                    acknowledged BOOLEAN
                )
            """)
    
    def save_alert(self, alert: Alert):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.timestamp.timestamp(),
                alert.severity.value,
                alert.alert_type.value,
                alert.title,
                alert.message,
                alert.risk_score,
                alert.confidence,
                json.dumps(alert.affected_models),
                json.dumps(alert.recommendations),
                alert.resolved,
                alert.acknowledged
            ))
    
    def get_active_alerts(self) -> List[Alert]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC
            """)
            return [self._row_to_alert(row) for row in cursor.fetchall()]
    
    def get_alerts_by_timeframe(self, hours: int = 24) -> List[Alert]:
        cutoff = datetime.now() - timedelta(hours=hours)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM alerts WHERE timestamp > ? ORDER BY timestamp DESC
            """, (cutoff.timestamp(),))
            return [self._row_to_alert(row) for row in cursor.fetchall()]
    
    def _row_to_alert(self, row) -> Alert:
        return Alert(
            id=row[0],
            timestamp=datetime.fromtimestamp(row[1]),
            severity=AlertSeverity(row[2]),
            alert_type=AlertType(row[3]),
            title=row[4],
            message=row[5],
            risk_score=row[6],
            confidence=row[7],
            affected_models=json.loads(row[8]),
            recommendations=json.loads(row[9]),
            resolved=bool(row[10]),
            acknowledged=bool(row[11])
        )

class NotificationManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhook', {})
        
    async def send_alert(self, alert: Alert):
        """Send alert through configured notification channels"""
        tasks = []
        
        if self.email_config.get('enabled', False):
            tasks.append(self._send_email_alert(alert))
        
        if self.webhook_config.get('enabled', False):
            tasks.append(self._send_webhook_alert(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"🚨 Supply Chain Alert: {alert.title}"
            
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls', True):
                server.starttls()
            if self.email_config.get('username'):
                server.login(self.email_config['username'], self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            logger.info(f"Email alert sent for {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification"""
        try:
            import aiohttp
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'risk_score': alert.risk_score,
                'confidence': alert.confidence
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_config['url'],
                    json=payload,
                    headers=self.webhook_config.get('headers', {})
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent for {alert.id}")
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format HTML email body"""
        severity_colors = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        return f"""
        <html>
        <body>
            <h2 style="color: {severity_colors[alert.severity]};">
                {alert.severity.value.upper()} Alert: {alert.title}
            </h2>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Risk Score:</strong> {alert.risk_score:.3f}</p>
            <p><strong>Confidence:</strong> {alert.confidence:.3f}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            
            <h3>Affected Models:</h3>
            <ul>
                {''.join(f'<li>{model}</li>' for model in alert.affected_models)}
            </ul>
            
            <h3>Recommendations:</h3>
            <ol>
                {''.join(f'<li>{rec}</li>' for rec in alert.recommendations)}
            </ol>
            
            <p><em>This is an automated alert from the Supply Chain Risk Intelligence System.</em></p>
        </body>
        </html>
        """

class RiskMonitor:
    def __init__(self, alert_db: AlertDatabase, notification_manager: NotificationManager):
        self.alert_db = alert_db
        self.notification_manager = notification_manager
        self.risk_history = []
        self.thresholds = {
            'critical_risk': 0.85,
            'high_risk': 0.70,
            'medium_risk': 0.50,
            'confidence_drop': 0.60,
            'trend_anomaly': 0.30
        }
    
    async def evaluate_risk(self, prediction_data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate risk data and generate alerts if necessary"""
        risk_score = prediction_data['risk_scores']['overall_risk']
        confidence = prediction_data['risk_scores']['confidence']
        timestamp = datetime.now()
        
        # Store risk history for trend analysis
        self.risk_history.append({
            'timestamp': timestamp,
            'risk_score': risk_score,
            'confidence': confidence
        })
        
        # Keep only last 24 hours of history
        cutoff = timestamp - timedelta(hours=24)
        self.risk_history = [
            h for h in self.risk_history 
            if h['timestamp'] > cutoff
        ]
        
        alert = None
        
        # Check critical risk threshold
        if risk_score >= self.thresholds['critical_risk']:
            alert = Alert(
                id=f"critical_{timestamp.timestamp()}",
                timestamp=timestamp,
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.RISK_THRESHOLD,
                title="Critical Supply Chain Risk Detected",
                message=f"Risk score has reached critical level: {risk_score:.3f}",
                risk_score=risk_score,
                confidence=confidence,
                affected_models=list(prediction_data['individual_predictions'].keys()),
                recommendations=[
                    "IMMEDIATE ACTION REQUIRED",
                    "Activate emergency response protocols",
                    "Contact all critical suppliers immediately",
                    "Prepare alternative sourcing options"
                ]
            )
        
        # Check high risk threshold
        elif risk_score >= self.thresholds['high_risk']:
            alert = Alert(
                id=f"high_{timestamp.timestamp()}",
                timestamp=timestamp,
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.RISK_THRESHOLD,
                title="High Supply Chain Risk Detected",
                message=f"Risk score has reached high level: {risk_score:.3f}",
                risk_score=risk_score,
                confidence=confidence,
                affected_models=list(prediction_data['individual_predictions'].keys()),
                recommendations=[
                    "Increase monitoring frequency",
                    "Review contingency plans",
                    "Prepare stakeholder communications"
                ]
            )
        
        # Check confidence drop
        elif confidence < self.thresholds['confidence_drop']:
            alert = Alert(
                id=f"confidence_{timestamp.timestamp()}",
                timestamp=timestamp,
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.CONFIDENCE_DROP,
                title="Model Confidence Drop Detected",
                message=f"Prediction confidence has dropped to {confidence:.3f}",
                risk_score=risk_score,
                confidence=confidence,
                affected_models=list(prediction_data['individual_predictions'].keys()),
                recommendations=[
                    "Review model performance",
                    "Check data quality",
                    "Consider model retraining"
                ]
            )
        
        # Check for trend anomalies
        if len(self.risk_history) >= 5:
            trend_alert = self._check_trend_anomaly(risk_score, confidence, timestamp)
            if trend_alert:
                alert = trend_alert
        
        if alert:
            self.alert_db.save_alert(alert)
            await self.notification_manager.send_alert(alert)
            logger.warning(f"Alert generated: {alert.title}")
        
        return alert
    
    def _check_trend_anomaly(self, current_risk: float, current_confidence: float, 
                           timestamp: datetime) -> Optional[Alert]:
        """Check for anomalous trends in risk patterns"""
        recent_risks = [h['risk_score'] for h in self.risk_history[-5:]]
        avg_recent_risk = sum(recent_risks) / len(recent_risks)
        
        # Check for sudden spike
        if current_risk > avg_recent_risk + self.thresholds['trend_anomaly']:
            return Alert(
                id=f"anomaly_{timestamp.timestamp()}",
                timestamp=timestamp,
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.TREND_ANOMALY,
                title="Risk Trend Anomaly Detected",
                message=f"Sudden risk increase detected: {current_risk:.3f} (avg: {avg_recent_risk:.3f})",
                risk_score=current_risk,
                confidence=current_confidence,
                affected_models=[],
                recommendations=[
                    "Investigate recent changes",
                    "Review data sources",
                    "Check for external events"
                ]
            )
        
        return None

class AlertSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_db = AlertDatabase(config.get('database_path', 'data/alerts.db'))
        self.notification_manager = NotificationManager(config.get('notifications', {}))
        self.risk_monitor = RiskMonitor(self.alert_db, self.notification_manager)
        self.running = False
    
    async def start_monitoring(self):
        """Start the alert monitoring system"""
        self.running = True
        logger.info("Alert system started")
    
    async def stop_monitoring(self):
        """Stop the alert monitoring system"""
        self.running = False
        logger.info("Alert system stopped")
    
    async def process_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Alert]:
        """Process a risk prediction and check for alerts"""
        if not self.running:
            return None
        
        return await self.risk_monitor.evaluate_risk(prediction_data)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return self.alert_db.get_active_alerts()
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified timeframe"""
        return self.alert_db.get_alerts_by_timeframe(hours)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        # Implementation for acknowledging alerts
        pass
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        # Implementation for resolving alerts
        pass

# Example configuration
DEFAULT_CONFIG = {
    'database_path': 'data/alerts.db',
    'notifications': {
        'email': {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from': 'alerts@company.com',
            'recipients': ['admin@company.com'],
            'username': '',
            'password': '',
            'use_tls': True
        },
        'webhook': {
            'enabled': False,
            'url': 'https://hooks.slack.com/services/...',
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    }
}
