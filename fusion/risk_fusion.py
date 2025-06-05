import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskFusionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_model = MultiModalTransformer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        self.risk_weights = {
            'satellite': 0.25,
            'sentiment': 0.20,
            'weather': 0.15,
            'economic': 0.25,
            'news': 0.15
        }
        
    def fuse_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        features = self._extract_features(predictions)
        fused_output = self.fusion_model(features)
        risk_scores = self._calculate_risk_scores(fused_output)
        return risk_scores
        
    def _extract_features(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_list = []
        for modality, pred in predictions.items():
            if pred.dim() > 1:
                features = pred.mean(dim=1) if pred.dim() > 2 else pred
            else:
                features = pred.unsqueeze(0) if pred.dim() == 1 else pred
            feature_list.append(features)
        return torch.cat(feature_list, dim=-1)
        
    def _calculate_risk_scores(self, fused_output: torch.Tensor) -> Dict[str, float]:
        risk_prob = torch.sigmoid(fused_output).item()
        
        return {
            'overall_risk': risk_prob,
            'operational_risk': min(risk_prob * 1.2, 1.0),
            'financial_risk': risk_prob * 0.8,
            'reputational_risk': risk_prob * 0.6,
            'confidence': self._calculate_confidence(risk_prob)
        }
        
    def _calculate_confidence(self, risk_score: float) -> float:
        return 1.0 - abs(0.5 - risk_score) * 2
        
    def train_fusion_model(self, training_data: Dict[str, Any]) -> None:
        logger.info("Training fusion model")
        optimizer = torch.optim.Adam(
            self.fusion_model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        self.fusion_model.train()
        for epoch in range(50):
            synthetic_features = torch.randn(32, 512)
            synthetic_labels = torch.randint(0, 2, (32,)).float()
            
            outputs = self.fusion_model(synthetic_features)
            loss = criterion(outputs.squeeze(), synthetic_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        logger.info("Completed fusion model training")

class MultiModalTransformer(nn.Module):
    def __init__(self, hidden_size: int = 512, num_attention_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.feature_projection = nn.Linear(1024, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = features.shape[0], 1
        
        if features.dim() == 2 and features.shape[1] != self.hidden_size:
            features = self.feature_projection(features)
            
        if features.dim() == 2:
            features = features.unsqueeze(1)
            
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        features = features + pos_encoding
        
        transformer_output = self.transformer(features)
        pooled_output = transformer_output.mean(dim=1)
        risk_score = self.risk_head(pooled_output)
        
        return risk_score

class AttentionVisualizer:
    def __init__(self, fusion_model: MultiModalTransformer):
        self.model = fusion_model
        
    def visualize_attention_weights(self, features: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            attention_weights = []
            for layer in self.model.transformer.layers:
                attention_weights.append(layer.self_attn(features, features, features)[1])
        return torch.stack(attention_weights).numpy()
