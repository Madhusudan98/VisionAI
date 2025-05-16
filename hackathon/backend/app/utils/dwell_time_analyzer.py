"""
Dwell Time Analyzer

A specialized module that analyzes how long objects stay in specific areas.
This module has a single responsibility: track and analyze dwell times.
"""

import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DwellTimeAnalyzer:
    """
    Specialized analyzer for tracking how long objects dwell in areas.
    Only responsible for tracking dwell times and triggering alerts for unusual patterns.
    """
    
    def __init__(
        self,
        areas: Dict[str, np.ndarray],
        thresholds: Dict[str, Dict[str, float]] = None,
        data_dir: str = "data/dwell",
        save_interval: int = 300,  # Save data every 5 minutes
        callback: Optional[Callable] = None
    ):
        """
        Initialize the dwell time analyzer.
        
        Args:
            areas: Dictionary mapping area IDs to polygon coordinates (normalized 0-1)
            thresholds: Dictionary mapping area IDs to threshold values
                        Format: {area_id: {"short": seconds, "normal": seconds, "long": seconds}}
            data_dir: Directory to save dwell time data
            save_interval: How often to save data (in seconds)
            callback: Optional callback function called when thresholds are exceeded
        """
        self.areas = areas
        self.data_dir = data_dir
        self.save_interval = save_interval
        self.callback = callback
        
        # Set default thresholds if not provided
        if thresholds is None:
            self.thresholds = {
                area_id: {
                    "short": 5.0,    # 5 seconds is unusually short
                    "normal": 60.0,  # 1 minute is normal
                    "long": 180.0    # 3 minutes is unusually long
                } for area_id in areas
            }
        else:
            self.thresholds = thresholds
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Track object dwell times
        # {object_id: {area_id: {"entry_time": timestamp, "total_time": seconds, "visits": count, "inside": bool}}}
        self.dwell_data = {}
        
        # Track dwell time alerts
        self.dwell_alerts = []
        
        # Track last save time
        self.last_save_time = time.time()
        
        # Load previous data if available
        self._load_data()
        
        logger.info(f"Dwell time analyzer initialized with {len(areas)} areas")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> Optional[Dict]:
        """
        Update object position and check dwell times.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with alert details if threshold exceeded, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize object tracking if new
        if object_id not in self.dwell_data:
            self.dwell_data[object_id] = {}
        
        # Check each area
        for area_id, area_polygon in self.areas.items():
            # Determine if object is inside this area
            is_inside = self._is_in_polygon(position, area_polygon)
            
            # Initialize area tracking for this object if needed
            if area_id not in self.dwell_data[object_id]:
                self.dwell_data[object_id][area_id] = {
                    "entry_time": timestamp if is_inside else None,
                    "total_time": 0.0,
                    "visits": 0,
                    "inside": is_inside
                }
                
                # Count as a visit if entering
                if is_inside:
                    self.dwell_data[object_id][area_id]["visits"] += 1
                
                continue
            
            # Get previous state
            area_data = self.dwell_data[object_id][area_id]
            prev_inside = area_data["inside"]
            
            # Check for state change
            if is_inside != prev_inside:
                if is_inside:
                    # Object entered area
                    area_data["entry_time"] = timestamp
                    area_data["visits"] += 1
                    logger.debug(f"Object {object_id} entered area {area_id}")
                else:
                    # Object exited area
                    if area_data["entry_time"] is not None:
                        # Calculate dwell time for this visit
                        dwell_time = timestamp - area_data["entry_time"]
                        area_data["total_time"] += dwell_time
                        
                        # Check if dwell time is unusual
                        thresholds = self.thresholds.get(area_id, {
                            "short": 5.0, "normal": 60.0, "long": 180.0
                        })
                        
                        alert = None
                        if dwell_time < thresholds["short"]:
                            alert = self._create_alert(
                                object_id, area_id, timestamp, dwell_time, "short",
                                f"Object {object_id} spent only {dwell_time:.1f}s in area {area_id}"
                            )
                        elif dwell_time > thresholds["long"]:
                            alert = self._create_alert(
                                object_id, area_id, timestamp, dwell_time, "long",
                                f"Object {object_id} spent {dwell_time:.1f}s in area {area_id}"
                            )
                        
                        if alert:
                            return alert
                        
                        logger.debug(f"Object {object_id} exited area {area_id} after {dwell_time:.1f}s")
            
            # Update state
            area_data["inside"] = is_inside
        
        # Check for ongoing long dwell times
        for area_id, area_data in self.dwell_data[object_id].items():
            if area_data["inside"] and area_data["entry_time"] is not None:
                current_dwell = timestamp - area_data["entry_time"]
                thresholds = self.thresholds.get(area_id, {
                    "short": 5.0, "normal": 60.0, "long": 180.0
                })
                
                # Check if current dwell time exceeds long threshold
                if current_dwell > thresholds["long"]:
                    # Only alert if we haven't already alerted for this dwell period
                    # We'll use a simple approach: check if the last alert for this object/area
                    # was within the last minute
                    should_alert = True
                    for alert in reversed(self.dwell_alerts):
                        if (alert["object_id"] == object_id and 
                                alert["area_id"] == area_id and
                                timestamp - alert["timestamp"] < 60.0):
                            should_alert = False
                            break
                    
                    if should_alert:
                        return self._create_alert(
                            object_id, area_id, timestamp, current_dwell, "ongoing_long",
                            f"Object {object_id} has been in area {area_id} for {current_dwell:.1f}s"
                        )
        
        # Save data if interval has passed
        if timestamp - self.last_save_time >= self.save_interval:
            self._save_data()
        
        return None
    
    def get_dwell_stats(self, object_id: Optional[int] = None, area_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dwell time statistics.
        
        Args:
            object_id: Optional filter by object ID
            area_id: Optional filter by area ID
            
        Returns:
            Dictionary with dwell time statistics
        """
        stats = {
            "timestamp": time.time(),
            "areas": {},
            "objects": {}
        }
        
        # If specific object requested
        if object_id is not None:
            if object_id in self.dwell_data:
                stats["objects"][object_id] = self.dwell_data[object_id]
            return stats
        
        # If specific area requested
        if area_id is not None:
            for obj_id, areas in self.dwell_data.items():
                if area_id in areas:
                    if obj_id not in stats["objects"]:
                        stats["objects"][obj_id] = {}
                    stats["objects"][obj_id][area_id] = areas[area_id]
            return stats
        
        # Compile area statistics
        for obj_id, areas in self.dwell_data.items():
            for a_id, data in areas.items():
                if a_id not in stats["areas"]:
                    stats["areas"][a_id] = {
                        "total_visits": 0,
                        "total_dwell_time": 0.0,
                        "avg_dwell_time": 0.0,
                        "current_visitors": 0
                    }
                
                stats["areas"][a_id]["total_visits"] += data["visits"]
                stats["areas"][a_id]["total_dwell_time"] += data["total_time"]
                if data["inside"]:
                    stats["areas"][a_id]["current_visitors"] += 1
        
        # Calculate averages
        for a_id, data in stats["areas"].items():
            if data["total_visits"] > 0:
                data["avg_dwell_time"] = data["total_dwell_time"] / data["total_visits"]
        
        return stats
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent dwell time alerts.
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.dwell_alerts[-count:] if self.dwell_alerts else []
    
    def reset(self):
        """Reset the analyzer, clearing all tracking data but keeping areas and thresholds."""
        # Save current data first
        self._save_data()
        
        # Reset tracking data
        self.dwell_data = {}
        
        logger.info("Dwell time analyzer reset")
    
    def _create_alert(self, object_id: int, area_id: str, timestamp: float, 
                     dwell_time: float, alert_type: str, message: str) -> Dict[str, Any]:
        """Create a dwell time alert."""
        alert = {
            "object_id": object_id,
            "area_id": area_id,
            "timestamp": timestamp,
            "dwell_time": dwell_time,
            "alert_type": alert_type,
            "message": message
        }
        
        # Add to alerts list
        self.dwell_alerts.append(alert)
        
        # Call callback if provided
        if self.callback:
            self.callback(alert)
        
        logger.info(f"Dwell time alert: {message}")
        return alert
    
    def _is_in_polygon(self, point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            point: The point to check (x, y)
            polygon: Array of polygon vertices
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _save_data(self):
        """Save dwell time data to file."""
        # Create a serializable version of the data
        save_data = {
            "timestamp": time.time(),
            "dwell_data": {},
            "thresholds": self.thresholds
        }
        
        # Convert object IDs to strings for JSON
        for obj_id, areas in self.dwell_data.items():
            save_data["dwell_data"][str(obj_id)] = areas
        
        # Create filename with timestamp
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(self.data_dir, f"dwell_times_{date_str}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Dwell time data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving dwell time data: {e}")
    
    def _load_data(self):
        """Load dwell time data from file if available."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(self.data_dir, f"dwell_times_{date_str}.json")
        
        if not os.path.exists(filename):
            logger.info(f"No previous dwell time data found for today at {filename}")
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert string object IDs back to integers
            for obj_id_str, areas in data.get("dwell_data", {}).items():
                self.dwell_data[int(obj_id_str)] = areas
            
            # Load thresholds if available
            if "thresholds" in data:
                self.thresholds.update(data["thresholds"])
            
            logger.info(f"Loaded dwell time data from {filename}")
        except Exception as e:
            logger.error(f"Error loading dwell time data: {e}")
    
    def draw_areas(self, frame: np.ndarray, colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw monitored areas on a frame.
        
        Args:
            frame: The frame to draw on
            colors: Dictionary mapping area IDs to RGB color tuples
            
        Returns:
            Frame with areas drawn
        """
        if colors is None:
            # Default colors if not provided
            colors = {
                area_id: (0, 0, 255) for area_id in self.areas
            }
        
        height, width = frame.shape[:2]
        
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        
        for area_id, polygon in self.areas.items():
            color = colors.get(area_id, (0, 0, 255))
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for x, y in polygon:
                points.append((int(x * width), int(y * height)))
            
            # Draw polygon with some transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(points)], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw outline
            cv2.polylines(frame, [np.array(points)], True, color, 2)
            
            # Calculate centroid for label
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Draw area ID
            cv2.putText(frame, area_id, (int(centroid_x), int(centroid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame 