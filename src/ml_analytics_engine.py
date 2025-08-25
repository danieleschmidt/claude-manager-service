#!/usr/bin/env python3
"""
MACHINE LEARNING ANALYTICS ENGINE - Generation 3
Advanced analytics, predictive modeling, and intelligent insights
"""

import asyncio
import json
import time
import math
import statistics
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import pickle
import joblib

from src.logger import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


class PredictionType(Enum):
    """Types of predictions"""
    LOAD_PREDICTION = "load_prediction"
    PERFORMANCE_FORECAST = "performance_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    SCALING_RECOMMENDATION = "scaling_recommendation"
    FAILURE_PREDICTION = "failure_prediction"


@dataclass
class AnalyticsMetric:
    """Analytics metric data point"""
    timestamp: float
    metric_name: str
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_feature_vector(self, feature_names: List[str]) -> List[float]:
        """Convert to feature vector for ML models"""
        features = []
        
        for feature_name in feature_names:
            if feature_name == "timestamp":
                features.append(self.timestamp)
            elif feature_name == "value":
                features.append(self.value)
            elif feature_name in self.context:
                value = self.context[feature_name]
                if isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)  # Default for non-numeric values
            else:
                features.append(0.0)  # Default for missing features
        
        return features


@dataclass
class PredictionResult:
    """ML prediction result"""
    prediction_type: PredictionType
    predicted_value: float
    confidence_score: float
    prediction_timestamp: float
    model_used: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    explanation: str = ""


class SimpleMLModel:
    """Simplified ML model implementation (using basic algorithms)"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.is_trained = False
        self.feature_names: List[str] = []
        self.model_params = {}
        self.training_data = []
        self.training_labels = []
        self.feature_importance = {}
        
    def train(self, training_data: List[List[float]], labels: List[float], feature_names: List[str]):
        """Train the model with given data"""
        self.feature_names = feature_names
        self.training_data = training_data
        self.training_labels = labels
        
        try:
            if self.model_type == ModelType.LINEAR_REGRESSION:
                self._train_linear_regression()
            elif self.model_type == ModelType.TIME_SERIES:
                self._train_time_series()
            elif self.model_type == ModelType.ANOMALY_DETECTION:
                self._train_anomaly_detection()
            else:
                # Default simple model
                self._train_simple_average()
            
            self.is_trained = True
            logger.info(f"Model {self.model_type.value} trained with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.is_trained = False
    
    def _train_linear_regression(self):
        """Simple linear regression implementation"""
        if len(self.training_data) < 2:
            raise ValueError("Need at least 2 data points for linear regression")
        
        # Simple single-feature linear regression for now
        X = [row[0] if row else 0 for row in self.training_data]  # First feature
        y = self.training_labels
        
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_xy = sum(x * y for x, y in zip(X, y))
        sum_xx = sum(x * x for x in X)
        
        # Calculate slope and intercept
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            self.model_params = {"slope": 0, "intercept": statistics.mean(y)}
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            self.model_params = {"slope": slope, "intercept": intercept}
        
        # Feature importance (simplified)
        self.feature_importance = {
            self.feature_names[0] if self.feature_names else "feature_0": abs(self.model_params["slope"])
        }
    
    def _train_time_series(self):
        """Simple time series model (moving average with trend)"""
        if len(self.training_labels) < 5:
            raise ValueError("Need at least 5 data points for time series")
        
        # Calculate trend
        values = self.training_labels
        n = len(values)
        
        # Simple linear trend calculation
        x_vals = list(range(n))
        trend_slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x_vals) * sum(values)) / (n * sum(i * i for i in x_vals) - sum(x_vals) ** 2)
        
        # Moving average window
        window_size = min(5, n // 2)
        recent_avg = statistics.mean(values[-window_size:])
        
        self.model_params = {
            "trend_slope": trend_slope,
            "recent_average": recent_avg,
            "window_size": window_size
        }
        
        # Feature importance
        self.feature_importance = {
            "trend": abs(trend_slope),
            "recent_average": 1.0
        }
    
    def _train_anomaly_detection(self):
        """Simple anomaly detection based on statistical bounds"""
        if len(self.training_labels) < 10:
            raise ValueError("Need at least 10 data points for anomaly detection")
        
        values = self.training_labels
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate percentiles
        sorted_vals = sorted(values)
        p25 = sorted_vals[len(sorted_vals) // 4]
        p75 = sorted_vals[3 * len(sorted_vals) // 4]
        iqr = p75 - p25
        
        self.model_params = {
            "mean": mean_val,
            "std": std_val,
            "p25": p25,
            "p75": p75,
            "iqr": iqr,
            "lower_bound": p25 - 1.5 * iqr,
            "upper_bound": p75 + 1.5 * iqr
        }
        
        self.feature_importance = {"statistical_bounds": 1.0}
    
    def _train_simple_average(self):
        """Fallback simple averaging model"""
        if not self.training_labels:
            raise ValueError("No training data provided")
        
        self.model_params = {
            "average": statistics.mean(self.training_labels),
            "std": statistics.stdev(self.training_labels) if len(self.training_labels) > 1 else 0.0
        }
        
        self.feature_importance = {"average": 1.0}
    
    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Make prediction with confidence score"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            if self.model_type == ModelType.LINEAR_REGRESSION:
                return self._predict_linear_regression(features)
            elif self.model_type == ModelType.TIME_SERIES:
                return self._predict_time_series(features)
            elif self.model_type == ModelType.ANOMALY_DETECTION:
                return self._predict_anomaly(features)
            else:
                return self._predict_simple_average(features)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, 0.0
    
    def _predict_linear_regression(self, features: List[float]) -> Tuple[float, float]:
        """Linear regression prediction"""
        if not features:
            return 0.0, 0.0
        
        x = features[0]
        prediction = self.model_params["slope"] * x + self.model_params["intercept"]
        
        # Simple confidence based on how close x is to training data range
        if self.training_data:
            training_x = [row[0] if row else 0 for row in self.training_data]
            min_x, max_x = min(training_x), max(training_x)
            
            if min_x <= x <= max_x:
                confidence = 0.8  # High confidence within training range
            else:
                # Lower confidence for extrapolation
                distance = min(abs(x - min_x), abs(x - max_x))
                max_distance = max_x - min_x if max_x != min_x else 1
                confidence = max(0.3, 0.8 - (distance / max_distance))
        else:
            confidence = 0.5
        
        return prediction, confidence
    
    def _predict_time_series(self, features: List[float]) -> Tuple[float, float]:
        """Time series prediction"""
        trend = self.model_params["trend_slope"]
        recent_avg = self.model_params["recent_average"]
        
        # Simple prediction: recent average + trend
        prediction = recent_avg + trend
        confidence = 0.7  # Medium confidence for time series
        
        return prediction, confidence
    
    def _predict_anomaly(self, features: List[float]) -> Tuple[float, float]:
        """Anomaly detection prediction (returns anomaly score)"""
        if not features:
            return 0.0, 0.0
        
        value = features[0]
        lower_bound = self.model_params["lower_bound"]
        upper_bound = self.model_params["upper_bound"]
        mean_val = self.model_params["mean"]
        std_val = self.model_params["std"]
        
        # Calculate anomaly score
        if lower_bound <= value <= upper_bound:
            # Normal range
            anomaly_score = 0.0
        else:
            # Anomaly - calculate severity
            if value < lower_bound:
                anomaly_score = (lower_bound - value) / (std_val + 1e-6)
            else:
                anomaly_score = (value - upper_bound) / (std_val + 1e-6)
            
            anomaly_score = min(anomaly_score, 10.0)  # Cap at 10
        
        confidence = 0.9 if anomaly_score > 0 else 0.7
        
        return anomaly_score, confidence
    
    def _predict_simple_average(self, features: List[float]) -> Tuple[float, float]:
        """Simple average prediction"""
        prediction = self.model_params["average"]
        confidence = 0.6  # Medium confidence for simple model
        
        return prediction, confidence


class MLAnalyticsEngine:
    """Advanced ML-powered analytics engine"""
    
    def __init__(self, 
                 max_history_size: int = 10000,
                 model_retrain_interval: float = 3600.0,  # 1 hour
                 min_training_samples: int = 50):
        
        self.max_history_size = max_history_size
        self.model_retrain_interval = model_retrain_interval
        self.min_training_samples = min_training_samples
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.prediction_history: deque = deque(maxlen=1000)
        
        # ML models
        self.models: Dict[PredictionType, SimpleMLModel] = {}
        self.last_training_time = {}
        
        # Feature engineering
        self.feature_extractors = {}
        self.feature_scalers = {}
        
        # Analytics state
        self.analytics_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "model_accuracy": defaultdict(float),
            "feature_importance": defaultdict(dict)
        }
        
        self._lock = threading.RLock()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different prediction types"""
        model_configs = {
            PredictionType.LOAD_PREDICTION: ModelType.TIME_SERIES,
            PredictionType.PERFORMANCE_FORECAST: ModelType.LINEAR_REGRESSION,
            PredictionType.ANOMALY_DETECTION: ModelType.ANOMALY_DETECTION,
            PredictionType.RESOURCE_OPTIMIZATION: ModelType.LINEAR_REGRESSION,
            PredictionType.SCALING_RECOMMENDATION: ModelType.TIME_SERIES,
            PredictionType.FAILURE_PREDICTION: ModelType.ANOMALY_DETECTION
        }
        
        for prediction_type, model_type in model_configs.items():
            self.models[prediction_type] = SimpleMLModel(model_type)
            self.last_training_time[prediction_type] = 0.0
            logger.info(f"Initialized {model_type.value} model for {prediction_type.value}")
    
    def add_metric(self, metric: AnalyticsMetric):
        """Add new metric data point"""
        with self._lock:
            self.metrics_history.append(metric)
            
            # Auto-retrain models if needed
            self._check_and_retrain_models()
    
    def add_metrics_batch(self, metrics: List[AnalyticsMetric]):
        """Add batch of metrics efficiently"""
        with self._lock:
            for metric in metrics:
                self.metrics_history.append(metric)
            
            self._check_and_retrain_models()
    
    def predict(self, 
                prediction_type: PredictionType,
                context: Optional[Dict[str, Any]] = None,
                forecast_horizon: int = 1) -> PredictionResult:
        """Make ML prediction"""
        
        with self._lock:
            model = self.models.get(prediction_type)
            if not model or not model.is_trained:
                # Train model if not trained yet
                self._train_model(prediction_type)
                model = self.models.get(prediction_type)
            
            if not model or not model.is_trained:
                # Fallback prediction
                return self._fallback_prediction(prediction_type)
            
            try:
                # Extract features for prediction
                features = self._extract_prediction_features(prediction_type, context)
                
                # Make prediction
                predicted_value, confidence_score = model.predict(features)
                
                # Create prediction result
                result = PredictionResult(
                    prediction_type=prediction_type,
                    predicted_value=predicted_value,
                    confidence_score=confidence_score,
                    prediction_timestamp=time.time(),
                    model_used=model.model_type.value,
                    feature_importance=model.feature_importance.copy(),
                    explanation=self._generate_prediction_explanation(prediction_type, predicted_value, confidence_score)
                )
                
                # Record prediction for analysis
                self.prediction_history.append(result)
                self.analytics_stats["total_predictions"] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction failed for {prediction_type.value}: {e}")
                return self._fallback_prediction(prediction_type)
    
    def _extract_prediction_features(self, 
                                   prediction_type: PredictionType,
                                   context: Optional[Dict[str, Any]]) -> List[float]:
        """Extract features for ML prediction"""
        
        features = []
        
        # Time-based features
        current_time = time.time()
        features.append(current_time)
        
        # Recent metrics features
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-20:]  # Last 20 metrics
            
            if prediction_type == PredictionType.LOAD_PREDICTION:
                # Load-related features
                load_values = [m.value for m in recent_metrics if "load" in m.metric_name.lower()]
                if load_values:
                    features.extend([
                        statistics.mean(load_values),
                        max(load_values),
                        min(load_values),
                        statistics.stdev(load_values) if len(load_values) > 1 else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                    
            elif prediction_type == PredictionType.PERFORMANCE_FORECAST:
                # Performance-related features
                perf_values = [m.value for m in recent_metrics if any(term in m.metric_name.lower() for term in ["response", "latency", "throughput"])]
                if perf_values:
                    features.extend([
                        statistics.mean(perf_values),
                        statistics.median(perf_values),
                        len([v for v in perf_values if v > statistics.mean(perf_values)])  # Count above average
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                    
            elif prediction_type == PredictionType.ANOMALY_DETECTION:
                # Anomaly detection features
                if recent_metrics:
                    latest_metric = recent_metrics[-1]
                    features.extend([latest_metric.value])
                else:
                    features.extend([0.0])
                    
            else:
                # Default features
                if recent_metrics:
                    recent_values = [m.value for m in recent_metrics]
                    features.extend([
                        statistics.mean(recent_values),
                        statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0])
        
        # Context features
        if context:
            for key in ["cpu_usage", "memory_usage", "request_rate", "error_rate"]:
                features.append(float(context.get(key, 0.0)))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _train_model(self, prediction_type: PredictionType):
        """Train ML model for specific prediction type"""
        
        if len(self.metrics_history) < self.min_training_samples:
            logger.warning(f"Not enough data to train {prediction_type.value} model")
            return
        
        try:
            # Prepare training data
            training_data, labels, feature_names = self._prepare_training_data(prediction_type)
            
            if not training_data or not labels:
                logger.warning(f"No training data available for {prediction_type.value}")
                return
            
            # Train model
            model = self.models[prediction_type]
            model.train(training_data, labels, feature_names)
            
            self.last_training_time[prediction_type] = time.time()
            
            # Update analytics stats
            self.analytics_stats["feature_importance"][prediction_type.value] = model.feature_importance.copy()
            
            logger.info(f"Successfully trained {prediction_type.value} model with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed for {prediction_type.value}: {e}")
    
    def _prepare_training_data(self, prediction_type: PredictionType) -> Tuple[List[List[float]], List[float], List[str]]:
        """Prepare training data for model"""
        
        training_data = []
        labels = []
        feature_names = ["timestamp", "mean_value", "max_value", "min_value", "std_value", "cpu_usage", "memory_usage", "request_rate", "error_rate"]
        
        # Convert metrics to training samples
        metrics_list = list(self.metrics_history)
        
        for i in range(len(metrics_list) - 1):  # Use next value as label
            current_metric = metrics_list[i]
            next_metric = metrics_list[i + 1]
            
            # Filter metrics based on prediction type
            if prediction_type == PredictionType.LOAD_PREDICTION and "load" not in current_metric.metric_name.lower():
                continue
            elif prediction_type == PredictionType.PERFORMANCE_FORECAST and not any(term in current_metric.metric_name.lower() for term in ["response", "latency", "throughput"]):
                continue
            
            # Extract features
            features = []
            features.append(current_metric.timestamp)
            
            # Get recent context for statistical features
            recent_window = metrics_list[max(0, i-10):i+1]
            recent_values = [m.value for m in recent_window]
            
            if recent_values:
                features.extend([
                    statistics.mean(recent_values),
                    max(recent_values),
                    min(recent_values),
                    statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Context features
            context = current_metric.context
            features.extend([
                float(context.get("cpu_usage", 0.0)),
                float(context.get("memory_usage", 0.0)),
                float(context.get("request_rate", 0.0)),
                float(context.get("error_rate", 0.0))
            ])
            
            training_data.append(features)
            labels.append(next_metric.value)
        
        return training_data, labels, feature_names
    
    def _check_and_retrain_models(self):
        """Check if models need retraining"""
        current_time = time.time()
        
        for prediction_type, last_train_time in self.last_training_time.items():
            if current_time - last_train_time > self.model_retrain_interval:
                logger.info(f"Retraining {prediction_type.value} model")
                self._train_model(prediction_type)
    
    def _fallback_prediction(self, prediction_type: PredictionType) -> PredictionResult:
        """Provide fallback prediction when ML model unavailable"""
        
        # Simple statistical fallback
        if self.metrics_history:
            recent_values = [m.value for m in list(self.metrics_history)[-10:]]
            predicted_value = statistics.mean(recent_values)
            confidence_score = 0.3  # Low confidence for fallback
        else:
            predicted_value = 0.0
            confidence_score = 0.1
        
        return PredictionResult(
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence_score=confidence_score,
            prediction_timestamp=time.time(),
            model_used="statistical_fallback",
            explanation="Fallback prediction using statistical methods"
        )
    
    def _generate_prediction_explanation(self, 
                                       prediction_type: PredictionType,
                                       predicted_value: float,
                                       confidence_score: float) -> str:
        """Generate human-readable explanation for prediction"""
        
        confidence_desc = "high" if confidence_score > 0.7 else "medium" if confidence_score > 0.4 else "low"
        
        explanations = {
            PredictionType.LOAD_PREDICTION: f"Predicted system load of {predicted_value:.2f} with {confidence_desc} confidence based on historical patterns",
            PredictionType.PERFORMANCE_FORECAST: f"Forecasted performance metric of {predicted_value:.2f} with {confidence_desc} confidence",
            PredictionType.ANOMALY_DETECTION: f"Anomaly score of {predicted_value:.2f} ({'anomalous' if predicted_value > 2.0 else 'normal'}) with {confidence_desc} confidence",
            PredictionType.RESOURCE_OPTIMIZATION: f"Recommended resource level of {predicted_value:.2f} with {confidence_desc} confidence",
            PredictionType.SCALING_RECOMMENDATION: f"Scaling recommendation score of {predicted_value:.2f} with {confidence_desc} confidence",
            PredictionType.FAILURE_PREDICTION: f"Failure risk score of {predicted_value:.2f} with {confidence_desc} confidence"
        }
        
        return explanations.get(prediction_type, f"Predicted value: {predicted_value:.2f}")
    
    def analyze_trends(self, metric_name: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze trends for specific metric"""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        relevant_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time and metric_name.lower() in m.metric_name.lower()
        ]
        
        if not relevant_metrics:
            return {"error": "No data found for analysis"}
        
        values = [m.value for m in relevant_metrics]
        timestamps = [m.timestamp for m in relevant_metrics]
        
        # Basic trend analysis
        if len(values) > 1:
            # Calculate trend slope
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * v for i, v in enumerate(values))
            sum_xx = sum(i * i for i in range(n))
            
            trend_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
            
            # Statistical measures
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            
            # Detect anomalies
            threshold = mean_val + 2 * std_val
            anomalies = [i for i, v in enumerate(values) if abs(v - mean_val) > 2 * std_val]
            
            return {
                "metric_name": metric_name,
                "time_window_hours": time_window_hours,
                "data_points": len(values),
                "trend_slope": trend_slope,
                "trend_direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable",
                "statistics": {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "range": max_val - min_val
                },
                "anomalies": {
                    "count": len(anomalies),
                    "percentage": (len(anomalies) / len(values)) * 100,
                    "indices": anomalies
                },
                "volatility": std_val / mean_val if mean_val != 0 else 0
            }
        else:
            return {"error": "Insufficient data for trend analysis"}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get ML model performance statistics"""
        
        performance_stats = {}
        
        for prediction_type, model in self.models.items():
            if model.is_trained:
                # Calculate accuracy from recent predictions
                relevant_predictions = [
                    p for p in self.prediction_history 
                    if p.prediction_type == prediction_type
                ]
                
                avg_confidence = statistics.mean([p.confidence_score for p in relevant_predictions]) if relevant_predictions else 0
                
                performance_stats[prediction_type.value] = {
                    "model_type": model.model_type.value,
                    "is_trained": model.is_trained,
                    "training_samples": len(model.training_data),
                    "recent_predictions": len(relevant_predictions),
                    "average_confidence": avg_confidence,
                    "feature_importance": model.feature_importance,
                    "last_training": self.last_training_time.get(prediction_type, 0)
                }
            else:
                performance_stats[prediction_type.value] = {
                    "model_type": "not_trained",
                    "is_trained": False
                }
        
        return {
            "model_performance": performance_stats,
            "overall_stats": self.analytics_stats,
            "total_metrics": len(self.metrics_history),
            "prediction_history": len(self.prediction_history)
        }
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate analytical insights from data"""
        
        insights = []
        
        if len(self.metrics_history) < 10:
            return [{"type": "warning", "message": "Insufficient data for meaningful insights"}]
        
        # Resource utilization insights
        cpu_metrics = [m for m in self.metrics_history if "cpu" in m.metric_name.lower()]
        if cpu_metrics:
            cpu_values = [m.value for m in cpu_metrics[-50:]]  # Last 50 CPU metrics
            avg_cpu = statistics.mean(cpu_values)
            
            if avg_cpu > 80:
                insights.append({
                    "type": "alert",
                    "category": "resource_utilization", 
                    "message": f"High CPU utilization detected (avg: {avg_cpu:.1f}%)",
                    "severity": "high",
                    "recommendation": "Consider scaling up or optimizing CPU-intensive processes"
                })
            elif avg_cpu < 20:
                insights.append({
                    "type": "optimization",
                    "category": "resource_utilization",
                    "message": f"Low CPU utilization detected (avg: {avg_cpu:.1f}%)",
                    "severity": "medium", 
                    "recommendation": "Consider scaling down to optimize costs"
                })
        
        # Performance trends
        response_metrics = [m for m in self.metrics_history if "response" in m.metric_name.lower() or "latency" in m.metric_name.lower()]
        if response_metrics:
            recent_response = [m.value for m in response_metrics[-20:]]
            trend_analysis = self.analyze_trends("response_time", 1)
            
            if "trend_direction" in trend_analysis and trend_analysis["trend_direction"] == "increasing":
                insights.append({
                    "type": "warning",
                    "category": "performance",
                    "message": "Response time trend is increasing",
                    "severity": "medium",
                    "recommendation": "Investigate performance bottlenecks and optimize critical paths"
                })
        
        # Error rate analysis
        error_metrics = [m for m in self.metrics_history if "error" in m.metric_name.lower()]
        if error_metrics:
            recent_errors = [m.value for m in error_metrics[-20:]]
            if recent_errors and statistics.mean(recent_errors) > 0.05:  # 5% error rate
                insights.append({
                    "type": "alert",
                    "category": "reliability",
                    "message": f"High error rate detected ({statistics.mean(recent_errors)*100:.2f}%)",
                    "severity": "high",
                    "recommendation": "Investigate error sources and implement error handling improvements"
                })
        
        # Predictive insights
        load_prediction = self.predict(PredictionType.LOAD_PREDICTION)
        if load_prediction.confidence_score > 0.6:
            if load_prediction.predicted_value > 0.8:
                insights.append({
                    "type": "prediction",
                    "category": "capacity_planning",
                    "message": f"Predicted high load ({load_prediction.predicted_value:.2f}) in near future",
                    "confidence": load_prediction.confidence_score,
                    "recommendation": "Prepare for potential scaling or load shedding"
                })
        
        return insights


# Factory functions
def create_ml_analytics_engine(**kwargs) -> MLAnalyticsEngine:
    """Create ML analytics engine"""
    return MLAnalyticsEngine(**kwargs)


# Demo
async def ml_analytics_demo():
    """Demonstration of ML analytics engine"""
    logger.info("Starting ML analytics engine demo")
    
    # Create analytics engine
    analytics = create_ml_analytics_engine(
        max_history_size=1000,
        min_training_samples=20
    )
    
    # Generate sample metrics
    logger.info("Generating sample metrics...")
    
    base_time = time.time() - 3600  # Start 1 hour ago
    
    for i in range(200):
        timestamp = base_time + (i * 18)  # Every 18 seconds
        
        # CPU usage metric
        cpu_value = 30 + 40 * math.sin(i * 0.1) + random.uniform(-5, 5)
        cpu_metric = AnalyticsMetric(
            timestamp=timestamp,
            metric_name="system.cpu_usage",
            value=max(0, min(100, cpu_value)),
            context={"host": "server1", "region": "us-west"},
            tags=["system", "performance"]
        )
        
        # Response time metric
        response_base = 100 + i * 0.5  # Gradual increase
        response_value = response_base + random.uniform(-20, 30)
        response_metric = AnalyticsMetric(
            timestamp=timestamp,
            metric_name="api.response_time",
            value=max(10, response_value),
            context={"endpoint": "/api/v1/data", "method": "GET"},
            tags=["api", "performance"]
        )
        
        # Load metric
        load_value = 0.3 + 0.4 * math.cos(i * 0.05) + random.uniform(-0.1, 0.1)
        load_metric = AnalyticsMetric(
            timestamp=timestamp,
            metric_name="system.load_average", 
            value=max(0, load_value),
            context={"cores": 8},
            tags=["system", "load"]
        )
        
        analytics.add_metric(cpu_metric)
        analytics.add_metric(response_metric)
        analytics.add_metric(load_metric)
    
    logger.info(f"Added {len(analytics.metrics_history)} metrics")
    
    # Test predictions
    logger.info("Testing ML predictions...")
    
    prediction_types = [
        PredictionType.LOAD_PREDICTION,
        PredictionType.PERFORMANCE_FORECAST,
        PredictionType.ANOMALY_DETECTION
    ]
    
    for pred_type in prediction_types:
        result = analytics.predict(pred_type, {"cpu_usage": 65.0, "memory_usage": 45.0})
        logger.info(f"{pred_type.value}: {result.predicted_value:.2f} (confidence: {result.confidence_score:.2f})")
        logger.info(f"  Explanation: {result.explanation}")
    
    # Trend analysis
    logger.info("Performing trend analysis...")
    cpu_trends = analytics.analyze_trends("cpu_usage", 1)
    logger.info(f"CPU trend: {cpu_trends.get('trend_direction', 'unknown')} (slope: {cpu_trends.get('trend_slope', 0):.4f})")
    
    # Model performance
    performance = analytics.get_model_performance()
    logger.info(f"Models trained: {sum(1 for p in performance['model_performance'].values() if p.get('is_trained', False))}")
    
    # Generate insights
    insights = analytics.generate_insights()
    logger.info(f"Generated {len(insights)} insights:")
    for insight in insights:
        logger.info(f"  {insight['type']}: {insight['message']}")
    
    logger.info("ML analytics engine demo completed")


if __name__ == "__main__":
    # Add random import for demo
    import random
    asyncio.run(ml_analytics_demo())