"""
Movement Anomaly Detector

A specialized module that detects unusual movement patterns.
This module has a single responsibility: identify anomalous movement behaviors.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from collections import deque
import json
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovementAnomalyDetector:
    """
    Specialized detector for unusual movement patterns.
    Only responsible for analyzing movement trajectories and detecting anomalies.
    """
    
    def __init__(
        self,
        history_size: int = 50,
        min_history_for_analysis: int = 10,
        suspicious_threshold: float = 0.7,
        data_dir: str = "data/anomalies",
        save_interval: int = 300,  # Save data every 5 minutes
        callback: Optional[Callable] = None
    ):
        """
        Initialize the movement anomaly detector.
        
        Args:
            history_size: Number of positions to keep in history for each object
            min_history_for_analysis: Minimum history points needed before analysis
            suspicious_threshold: Threshold for movement pattern to be considered suspicious
            data_dir: Directory to save anomaly detection data
            save_interval: How often to save data (in seconds)
            callback: Optional callback function called when anomalies are detected
        """
        self.history_size = history_size
        self.min_history_for_analysis = min_history_for_analysis
        self.suspicious_threshold = suspicious_threshold
        self.data_dir = data_dir
        self.save_interval = save_interval
        self.callback = callback
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Track position history for each object
        self.position_history = {}  # {object_id: deque([(position, timestamp), ...])}
        
        # Track movement metrics for each object
        self.movement_metrics = {}  # {object_id: {"speed": [], "direction_changes": [], ...}}
        
        # Track anomaly alerts
        self.anomaly_alerts = []
        
        # Track last save time
        self.last_save_time = time.time()
        
        logger.info("Movement anomaly detector initialized")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> Optional[Dict]:
        """
        Update object position and check for anomalous movement.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with anomaly details if detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize tracking if new object
        if object_id not in self.position_history:
            self.position_history[object_id] = deque(maxlen=self.history_size)
            self.movement_metrics[object_id] = {
                "speeds": [],
                "direction_changes": [],
                "path_efficiency": [],
                "last_anomaly_time": 0
            }
        
        # Add position to history
        self.position_history[object_id].append((position, timestamp))
        
        # Update movement metrics
        self._update_metrics(object_id)
        
        # Check for anomalies if we have enough history
        if len(self.position_history[object_id]) >= self.min_history_for_analysis:
            anomaly_score = self._analyze_movement(object_id)
            
            # Check if score exceeds threshold and we haven't alerted recently
            if (anomaly_score > self.suspicious_threshold and 
                    timestamp - self.movement_metrics[object_id]["last_anomaly_time"] > 10.0):
                
                # Update last anomaly time
                self.movement_metrics[object_id]["last_anomaly_time"] = timestamp
                
                # Create alert
                alert = self._create_alert(
                    object_id, timestamp, anomaly_score,
                    f"Unusual movement detected for object {object_id} (score: {anomaly_score:.2f})"
                )
                
                return alert
        
        # Save data if interval has passed
        if timestamp - self.last_save_time >= self.save_interval:
            self._save_data()
        
        return None
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent anomaly alerts.
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.anomaly_alerts[-count:] if self.anomaly_alerts else []
    
    def get_movement_metrics(self, object_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movement metrics for a specific object.
        
        Args:
            object_id: Unique ID of the object
            
        Returns:
            Dict with movement metrics or None if object not tracked
        """
        if object_id not in self.movement_metrics:
            return None
        
        metrics = self.movement_metrics[object_id].copy()
        
        # Add some derived metrics
        if metrics["speeds"]:
            metrics["avg_speed"] = sum(metrics["speeds"]) / len(metrics["speeds"])
            metrics["max_speed"] = max(metrics["speeds"])
        else:
            metrics["avg_speed"] = 0.0
            metrics["max_speed"] = 0.0
        
        if metrics["direction_changes"]:
            metrics["avg_direction_change"] = sum(metrics["direction_changes"]) / len(metrics["direction_changes"])
        else:
            metrics["avg_direction_change"] = 0.0
        
        if metrics["path_efficiency"]:
            metrics["avg_path_efficiency"] = sum(metrics["path_efficiency"]) / len(metrics["path_efficiency"])
        else:
            metrics["avg_path_efficiency"] = 1.0
        
        return metrics
    
    def reset(self):
        """Reset the detector, clearing all tracking data."""
        # Save current data first
        self._save_data()
        
        # Reset tracking data
        self.position_history = {}
        self.movement_metrics = {}
        
        logger.info("Movement anomaly detector reset")
    
    def _update_metrics(self, object_id: int):
        """Update movement metrics for an object."""
        history = self.position_history[object_id]
        metrics = self.movement_metrics[object_id]
        
        # Need at least 2 points to calculate metrics
        if len(history) < 2:
            return
        
        # Get the two most recent positions
        (pos2, time2) = history[-1]
        (pos1, time1) = history[-2]
        
        # Calculate speed (distance per second)
        distance = self._distance(pos1, pos2)
        time_diff = time2 - time1
        if time_diff > 0:
            speed = distance / time_diff
            metrics["speeds"].append(speed)
            
            # Keep only recent speed values (last 20)
            if len(metrics["speeds"]) > 20:
                metrics["speeds"] = metrics["speeds"][-20:]
        
        # Need at least 3 points to calculate direction changes
        if len(history) < 3:
            return
        
        # Get the three most recent positions
        (pos3, _) = history[-1]
        (pos2, _) = history[-2]
        (pos1, _) = history[-3]
        
        # Calculate direction change (angle between vectors)
        vec1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        vec2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
        
        # Avoid division by zero for stationary objects
        if self._magnitude(vec1) > 0.001 and self._magnitude(vec2) > 0.001:
            cos_angle = self._dot_product(vec1, vec2) / (self._magnitude(vec1) * self._magnitude(vec2))
            # Clamp to valid range for arccos
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle)
            metrics["direction_changes"].append(angle)
            
            # Keep only recent direction changes (last 20)
            if len(metrics["direction_changes"]) > 20:
                metrics["direction_changes"] = metrics["direction_changes"][-20:]
        
        # Calculate path efficiency (only if we have enough history)
        if len(history) >= 10:
            # Get the first and last positions in the last 10 points
            (first_pos, _) = history[-10]
            (last_pos, _) = history[-1]
            
            # Direct distance between first and last
            direct_distance = self._distance(first_pos, last_pos)
            
            # Actual path distance
            path_distance = 0.0
            for i in range(len(history) - 10, len(history) - 1):
                (pos1, _) = history[i]
                (pos2, _) = history[i + 1]
                path_distance += self._distance(pos1, pos2)
            
            # Efficiency is direct distance / path distance (1.0 is perfectly efficient)
            if path_distance > 0:
                efficiency = direct_distance / path_distance
                metrics["path_efficiency"].append(efficiency)
                
                # Keep only recent efficiency values (last 10)
                if len(metrics["path_efficiency"]) > 10:
                    metrics["path_efficiency"] = metrics["path_efficiency"][-10:]
    
    def _analyze_movement(self, object_id: int) -> float:
        """
        Analyze movement patterns and return an anomaly score.
        
        Args:
            object_id: Unique ID of the object
            
        Returns:
            Anomaly score (0.0 to 1.0, higher is more anomalous)
        """
        metrics = self.movement_metrics[object_id]
        
        # Initialize score components
        speed_score = 0.0
        direction_score = 0.0
        efficiency_score = 0.0
        
        # Calculate speed score (high variance in speed is suspicious)
        if len(metrics["speeds"]) >= 5:
            speeds = metrics["speeds"][-5:]
            mean_speed = sum(speeds) / len(speeds)
            if mean_speed > 0:
                # Calculate coefficient of variation (std / mean)
                variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
                std_dev = np.sqrt(variance)
                speed_score = min(1.0, std_dev / mean_speed)
        
        # Calculate direction score (frequent direction changes are suspicious)
        if len(metrics["direction_changes"]) >= 5:
            # Average direction change in radians (0 to pi)
            # Normalize to 0-1 range, with higher values being more suspicious
            avg_change = sum(metrics["direction_changes"][-5:]) / 5
            direction_score = min(1.0, avg_change / np.pi)
        
        # Calculate efficiency score (low efficiency is suspicious)
        if len(metrics["path_efficiency"]) >= 3:
            # Average efficiency (0 to 1, with 1 being most efficient)
            # Invert so higher values are more suspicious
            avg_efficiency = sum(metrics["path_efficiency"][-3:]) / 3
            efficiency_score = 1.0 - avg_efficiency
        
        # Combine scores with weights
        anomaly_score = (
            0.3 * speed_score +
            0.4 * direction_score +
            0.3 * efficiency_score
        )
        
        return anomaly_score
    
    def _create_alert(self, object_id: int, timestamp: float, 
                     anomaly_score: float, message: str) -> Dict[str, Any]:
        """Create a movement anomaly alert."""
        # Get recent trajectory for context
        trajectory = []
        for pos, ts in self.position_history[object_id]:
            trajectory.append({
                "position": pos,
                "timestamp": ts
            })
        
        alert = {
            "object_id": object_id,
            "timestamp": timestamp,
            "anomaly_score": anomaly_score,
            "message": message,
            "recent_trajectory": trajectory,
            "metrics": self.get_movement_metrics(object_id)
        }
        
        # Add to alerts list
        self.anomaly_alerts.append(alert)
        
        # Call callback if provided
        if self.callback:
            self.callback(alert)
        
        logger.info(f"Movement anomaly alert: {message}")
        return alert
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _magnitude(self, vector: Tuple[float, float]) -> float:
        """Calculate magnitude of a vector."""
        return np.sqrt(vector[0]**2 + vector[1]**2)
    
    def _dot_product(self, vec1: Tuple[float, float], vec2: Tuple[float, float]) -> float:
        """Calculate dot product of two vectors."""
        return vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    def _save_data(self):
        """Save anomaly detection data to file."""
        # Create a serializable version of the data (just save alerts)
        save_data = {
            "timestamp": time.time(),
            "anomaly_alerts": self.anomaly_alerts
        }
        
        # Create filename with timestamp
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(self.data_dir, f"movement_anomalies_{date_str}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Movement anomaly data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving movement anomaly data: {e}")
    
    def draw_trajectory(self, frame: np.ndarray, object_id: int, 
                       color: Tuple[int, int, int] = (255, 0, 0),
                       thickness: int = 2,
                       max_points: int = 30) -> np.ndarray:
        """
        Draw the trajectory of an object on a frame.
        
        Args:
            frame: The frame to draw on
            object_id: ID of the object to draw trajectory for
            color: RGB color tuple
            thickness: Line thickness
            max_points: Maximum number of trajectory points to draw
            
        Returns:
            Frame with trajectory drawn
        """
        if object_id not in self.position_history:
            return frame
        
        history = list(self.position_history[object_id])
        if len(history) < 2:
            return frame
        
        # Limit to max_points
        if len(history) > max_points:
            history = history[-max_points:]
        
        height, width = frame.shape[:2]
        
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        
        # Draw trajectory lines
        for i in range(len(history) - 1):
            pos1, _ = history[i]
            pos2, _ = history[i + 1]
            
            # Convert normalized coordinates to pixel coordinates
            p1 = (int(pos1[0] * width), int(pos1[1] * height))
            p2 = (int(pos2[0] * width), int(pos2[1] * height))
            
            # Draw line with increasing thickness for more recent points
            alpha = (i + 1) / len(history)  # 0 to 1
            line_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, p1, p2, line_color, thickness)
        
        # Draw dots at trajectory points
        for i, (pos, _) in enumerate(history):
            point = (int(pos[0] * width), int(pos[1] * height))
            
            # Size increases for more recent points
            point_size = int(2 + (i / len(history)) * 3)
            cv2.circle(frame, point, point_size, color, -1)
        
        return frame 