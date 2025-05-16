"""
Zone Detector

A specialized module that detects when objects enter or exit defined zones.
This module has a single responsibility: detect zone transitions and trigger events.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZoneDetector:
    """
    Specialized detector for zone entry/exit events.
    Only responsible for detecting when objects enter or exit defined zones.
    """
    
    def __init__(
        self,
        zones: Dict[str, np.ndarray],
        cooldown_period: float = 2.0,  # seconds between repeated triggers for same ID
        callback: Optional[Callable] = None
    ):
        """
        Initialize the zone detector.
        
        Args:
            zones: Dictionary mapping zone IDs to polygon coordinates (normalized 0-1)
            cooldown_period: Time in seconds before same object can trigger again for same zone
            callback: Optional callback function to call when zone transition occurs
        """
        self.zones = zones
        self.cooldown_period = cooldown_period
        self.callback = callback
        
        # Track object positions relative to zones
        self.object_zones = defaultdict(dict)  # {object_id: {zone_id: {"inside": bool, "last_transition": timestamp}}}
        
        # Track zone events
        self.zone_events = []
        
        logger.info(f"Zone detector initialized with {len(zones)} zones")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> Optional[Dict]:
        """
        Update object position and check if it entered or exited any zones.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with zone event details if transition occurred, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check each zone
        for zone_id, zone_polygon in self.zones.items():
            # Determine if object is inside this zone
            is_inside = self._is_in_polygon(position, zone_polygon)
            
            # Initialize zone tracking for this object if needed
            if zone_id not in self.object_zones[object_id]:
                self.object_zones[object_id][zone_id] = {
                    "inside": is_inside,
                    "last_transition": 0
                }
                continue
            
            # Get previous state
            prev_inside = self.object_zones[object_id][zone_id]["inside"]
            last_transition = self.object_zones[object_id][zone_id]["last_transition"]
            
            # Update current state
            self.object_zones[object_id][zone_id]["inside"] = is_inside
            
            # Check if state changed (zone transition)
            if is_inside != prev_inside:
                # Check if we're still in cooldown period
                if timestamp - last_transition < self.cooldown_period:
                    continue
                
                # Update last transition time
                self.object_zones[object_id][zone_id]["last_transition"] = timestamp
                
                # Determine transition type
                transition_type = "enter" if is_inside else "exit"
                
                # Create event
                event = {
                    "object_id": object_id,
                    "zone_id": zone_id,
                    "timestamp": timestamp,
                    "transition_type": transition_type,
                    "position": position
                }
                
                # Add to events list
                self.zone_events.append(event)
                
                # Call callback if provided
                if self.callback:
                    self.callback(event)
                
                logger.info(f"Object {object_id} {transition_type}ed zone {zone_id}")
                return event
        
        return None
    
    def is_object_in_zone(self, object_id: int, zone_id: str) -> bool:
        """
        Check if an object is currently in a specific zone.
        
        Args:
            object_id: Unique ID of the object
            zone_id: ID of the zone to check
            
        Returns:
            True if object is in zone, False otherwise
        """
        if object_id in self.object_zones and zone_id in self.object_zones[object_id]:
            return self.object_zones[object_id][zone_id]["inside"]
        return False
    
    def get_objects_in_zone(self, zone_id: str) -> List[int]:
        """
        Get all objects currently in a specific zone.
        
        Args:
            zone_id: ID of the zone to check
            
        Returns:
            List of object IDs in the zone
        """
        result = []
        for object_id, zones in self.object_zones.items():
            if zone_id in zones and zones[zone_id]["inside"]:
                result.append(object_id)
        return result
    
    def get_zones_for_object(self, object_id: int) -> List[str]:
        """
        Get all zones an object is currently in.
        
        Args:
            object_id: Unique ID of the object
            
        Returns:
            List of zone IDs the object is in
        """
        if object_id not in self.object_zones:
            return []
        
        return [zone_id for zone_id, data in self.object_zones[object_id].items() 
                if data["inside"]]
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent zone events.
        
        Args:
            count: Number of recent events to return
            
        Returns:
            List of recent zone events
        """
        return self.zone_events[-count:] if self.zone_events else []
    
    def reset(self):
        """Reset the detector, clearing all tracking data but keeping zones."""
        self.object_zones = defaultdict(dict)
        self.zone_events = []
        logger.info("Zone detector reset")
    
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
    
    def draw_zones(self, frame: np.ndarray, colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw zones on a frame.
        
        Args:
            frame: The frame to draw on
            colors: Dictionary mapping zone IDs to RGB color tuples
            
        Returns:
            Frame with zones drawn
        """
        if colors is None:
            # Default colors if not provided
            colors = {
                zone_id: (0, 255, 0) for zone_id in self.zones
            }
        
        height, width = frame.shape[:2]
        
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        
        for zone_id, polygon in self.zones.items():
            color = colors.get(zone_id, (0, 255, 0))
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for x, y in polygon:
                points.append((int(x * width), int(y * height)))
            
            # Draw polygon
            cv2.polylines(frame, [np.array(points)], True, color, 2)
            
            # Calculate centroid for label
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Draw zone ID
            cv2.putText(frame, zone_id, (int(centroid_x), int(centroid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame 