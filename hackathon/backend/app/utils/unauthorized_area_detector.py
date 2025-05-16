"""
Unauthorized Area Detector

A specialized module that detects when objects enter areas marked as unauthorized.
This module has a single responsibility: detect unauthorized area access and trigger alerts.
"""

import numpy as np
import time
import csv
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnauthorizedAreaDetector:
    """
    Specialized detector for unauthorized area access events.
    Only responsible for detecting when objects enter areas marked as unauthorized.
    """
    
    def __init__(
        self,
        labels_file: Optional[str] = None,
        areas: Optional[Dict[str, np.ndarray]] = None,
        cooldown_period: float = 5.0,  # seconds between repeated alerts for same ID
        data_dir: str = "data/unauthorized",
        callback: Optional[Callable] = None
    ):
        """
        Initialize the unauthorized area detector.
        
        Args:
            labels_file: Path to CSV file containing label data (optional)
            areas: Dictionary mapping area IDs to polygon coordinates (normalized 0-1) (optional)
            cooldown_period: Time in seconds before same object can trigger again
            data_dir: Directory to save alert data
            callback: Optional callback function to call when unauthorized access is detected
        """
        self.cooldown_period = cooldown_period
        self.data_dir = data_dir
        self.callback = callback
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize areas from labels file or directly provided areas
        self.areas = {}
        if labels_file:
            self._load_areas_from_labels(labels_file)
        elif areas:
            self.areas = areas
        
        # Track objects in unauthorized areas
        # {object_id: {"area_id": str, "last_alert_time": timestamp, "in_area": bool}}
        self.object_tracking = {}
        
        # Track unauthorized access events
        self.access_events = []
        
        logger.info(f"Unauthorized area detector initialized with {len(self.areas)} areas")
    
    def _load_areas_from_labels(self, labels_file: str):
        """
        Load unauthorized areas from a labels CSV file.
        
        Args:
            labels_file: Path to CSV file containing label data
        """
        if not os.path.exists(labels_file):
            logger.error(f"Labels file not found: {labels_file}")
            return
        
        try:
            with open(labels_file, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Extract label information
                    label_name = row.get('label_name', f"UnauthorizedArea_{i}")
                    
                    # Extract bounding box coordinates
                    try:
                        x = float(row.get('bbox_x', 0))
                        y = float(row.get('bbox_y', 0))
                        width = float(row.get('bbox_width', 0))
                        height = float(row.get('bbox_height', 0))
                        
                        # Get image dimensions for normalization
                        img_width = float(row.get('image_width', 1))
                        img_height = float(row.get('image_height', 1))
                        
                        # Normalize coordinates to 0-1 range
                        x_norm = x / img_width
                        y_norm = y / img_height
                        width_norm = width / img_width
                        height_norm = height / img_height
                        
                        # Create polygon from bounding box (clockwise from top-left)
                        polygon = np.array([
                            [x_norm, y_norm],                       # top-left
                            [x_norm + width_norm, y_norm],          # top-right
                            [x_norm + width_norm, y_norm + height_norm],  # bottom-right
                            [x_norm, y_norm + height_norm]          # bottom-left
                        ])
                        
                        # Add to areas dictionary
                        self.areas[label_name] = polygon
                        logger.info(f"Loaded unauthorized area '{label_name}' from labels file")
                    
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error parsing row {i} in labels file: {e}")
        
        except Exception as e:
            logger.error(f"Error loading labels file: {e}")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> Optional[Dict]:
        """
        Update object position and check if it entered an unauthorized area.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with event details if unauthorized access detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize object tracking if new
        if object_id not in self.object_tracking:
            self.object_tracking[object_id] = {}
        
        # Check each unauthorized area
        for area_id, polygon in self.areas.items():
            # Initialize area tracking for this object if needed
            if area_id not in self.object_tracking[object_id]:
                self.object_tracking[object_id][area_id] = {
                    "last_alert_time": 0,
                    "in_area": False
                }
            
            # Check if object is in this area
            is_in_area = self._is_in_polygon(position, polygon)
            area_data = self.object_tracking[object_id][area_id]
            
            # If object just entered the area, trigger alert
            if is_in_area and not area_data["in_area"]:
                # Check cooldown period
                if timestamp - area_data["last_alert_time"] >= self.cooldown_period:
                    # Update tracking data
                    area_data["in_area"] = True
                    area_data["last_alert_time"] = timestamp
                    
                    # Create event
                    event = self._create_event(object_id, area_id, position, timestamp)
                    return event
            
            # Update tracking state
            area_data["in_area"] = is_in_area
        
        return None
    
    def _create_event(self, object_id: int, area_id: str, position: Tuple[float, float], timestamp: float) -> Dict:
        """
        Create an unauthorized access event.
        
        Args:
            object_id: ID of the object
            area_id: ID of the unauthorized area
            position: Position of the object
            timestamp: Time of the event
            
        Returns:
            Event dictionary
        """
        event = {
            "type": "unauthorized_access",
            "object_id": object_id,
            "area_id": area_id,
            "position": position,
            "timestamp": timestamp,
            "message": f"Object {object_id} entered unauthorized area '{area_id}'"
        }
        
        # Add to events list
        self.access_events.append(event)
        
        # Call callback if provided
        if self.callback:
            self.callback(event)
        
        logger.warning(f"UNAUTHORIZED ACCESS: {event['message']}")
        return event
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent unauthorized access events.
        
        Args:
            count: Number of recent events to return
            
        Returns:
            List of recent events
        """
        return self.access_events[-count:] if self.access_events else []
    
    def is_object_in_unauthorized_area(self, object_id: int) -> bool:
        """
        Check if an object is currently in any unauthorized area.
        
        Args:
            object_id: ID of the object to check
            
        Returns:
            True if object is in any unauthorized area, False otherwise
        """
        if object_id not in self.object_tracking:
            return False
        
        for area_id, data in self.object_tracking[object_id].items():
            if data["in_area"]:
                return True
        
        return False
    
    def get_objects_in_unauthorized_areas(self) -> Dict[str, List[int]]:
        """
        Get all objects currently in unauthorized areas.
        
        Returns:
            Dictionary mapping area IDs to lists of object IDs
        """
        result = {area_id: [] for area_id in self.areas}
        
        for object_id, areas in self.object_tracking.items():
            for area_id, data in areas.items():
                if data["in_area"]:
                    result[area_id].append(object_id)
        
        return result
    
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
    
    def reset(self):
        """Reset the detector, clearing all tracking data but keeping areas."""
        self.object_tracking = {}
        self.access_events = []
        logger.info("Unauthorized area detector reset")
    
    def draw_areas(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), 
                  alpha: float = 0.3) -> np.ndarray:
        """
        Draw unauthorized areas on a frame.
        
        Args:
            frame: The frame to draw on
            color: RGB color tuple for unauthorized areas
            alpha: Transparency level for filled areas (0-1)
            
        Returns:
            Frame with areas drawn
        """
        height, width = frame.shape[:2]
        
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        
        for area_id, polygon in self.areas.items():
            # Convert normalized coordinates to pixel coordinates
            points = []
            for x, y in polygon:
                points.append((int(x * width), int(y * height)))
            
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(points)], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw outline
            cv2.polylines(frame, [np.array(points)], True, color, 2)
            
            # Calculate centroid for label
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Draw area ID
            cv2.putText(frame, f"UNAUTHORIZED: {area_id}", (int(centroid_x), int(centroid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame 