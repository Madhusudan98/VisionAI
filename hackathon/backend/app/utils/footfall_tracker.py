import cv2
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootfallTracker:
    """
    Class for tracking people entering and exiting defined zones.
    Useful for counting visitors in retail environments.
    """
    
    def __init__(
        self,
        roi_zones: Dict[int, np.ndarray],
        entry_roi_id: int,
        exit_roi_id: int,
        data_dir: str = "data/footfall",
        camera_id: Optional[int] = None,
        persistence_interval: int = 60  # Save data every 60 seconds
    ):
        """
        Initialize the FootfallTracker.
        
        Args:
            roi_zones: Dictionary mapping zone IDs to polygon coordinates (normalized 0-1)
            entry_roi_id: ID of the zone considered as entry
            exit_roi_id: ID of the zone considered as exit
            data_dir: Directory to save footfall data
            camera_id: Optional camera ID
            persistence_interval: How often to save data to disk (seconds)
        """
        self.roi_zones = roi_zones
        self.entry_roi_id = entry_roi_id
        self.exit_roi_id = exit_roi_id
        self.data_dir = data_dir
        self.camera_id = camera_id
        self.persistence_interval = persistence_interval
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Track people's positions
        self.tracked_people = {}  # {person_id: data}
        
        # Footfall statistics
        self.stats = {
            "total_entries": 0,
            "total_exits": 0,
            "current_visitors": 0,
            "hourly_entries": defaultdict(int),
            "hourly_exits": defaultdict(int),
            "daily_entries": defaultdict(int),
            "daily_exits": defaultdict(int)
        }
        
        # Track when people enter/exit zones
        self.zone_presence = {}  # {person_id: {zone_id: bool}}
        
        # Last time data was saved
        self.last_save_time = time.time()
        
        # Load previous stats if available
        self._load_stats()
        
        logger.info(f"FootfallTracker initialized with {len(roi_zones)} zones")
    
    def update_position(self, person_id: int, position: Tuple[float, float], timestamp: float = None):
        """
        Update a person's position and check for zone transitions.
        
        Args:
            person_id: Unique ID of the person
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize tracking if new person
        if person_id not in self.tracked_people:
            self.tracked_people[person_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "positions": [(position, timestamp)],
                "zone_transitions": [],
                "simple_transitions": []  # Added for simpler format
            }
            
            # Initialize zone presence
            self.zone_presence[person_id] = {}
            for zone_id in self.roi_zones:
                in_zone = self._is_in_polygon(position, self.roi_zones[zone_id])
                self.zone_presence[person_id][zone_id] = in_zone
        else:
            # Update existing person data
            person_data = self.tracked_people[person_id]
            
            # Update position history
            person_data["positions"].append((position, timestamp))
            person_data["last_seen"] = timestamp
            
            # Check for zone transitions
            for zone_id, zone_poly in self.roi_zones.items():
                in_zone = self._is_in_polygon(position, zone_poly)
                prev_in_zone = self.zone_presence[person_id].get(zone_id, False)
                
                if in_zone != prev_in_zone:
                    # Zone transition occurred
                    if in_zone:
                        # Person entered zone
                        transition_type = "enter"
                        if zone_id == self.entry_roi_id:
                            # Person entered the store
                            self._record_entry(person_id, timestamp)
                    else:
                        # Person exited zone
                        transition_type = "exit"
                        if zone_id == self.exit_roi_id:
                            # Person exited the store
                            self._record_exit(person_id, timestamp)
                    
                    # Record the transition
                    person_data["zone_transitions"].append({
                        "zone_id": zone_id,
                        "type": transition_type,
                        "timestamp": timestamp
                    })
                    
                    # Also store in simple format for compatibility
                    person_data["simple_transitions"].append((transition_type, timestamp))
                    
                    # Update zone presence
                    self.zone_presence[person_id][zone_id] = in_zone
        
        # Save stats periodically
        if timestamp - self.last_save_time > self.persistence_interval:
            self._save_stats()
            self.last_save_time = timestamp
    
    def _record_entry(self, person_id: int, timestamp: float):
        """Record a person entering the store"""
        self.stats["total_entries"] += 1
        self.stats["current_visitors"] += 1
        
        # Record hourly and daily stats
        dt = datetime.fromtimestamp(timestamp)
        hour_key = dt.strftime("%Y-%m-%d %H:00")
        day_key = dt.strftime("%Y-%m-%d")
        
        self.stats["hourly_entries"][hour_key] += 1
        self.stats["daily_entries"][day_key] += 1
        
        logger.info(f"Person {person_id} entered. Current visitors: {self.stats['current_visitors']}")
    
    def _record_exit(self, person_id: int, timestamp: float):
        """Record a person exiting the store"""
        self.stats["total_exits"] += 1
        self.stats["current_visitors"] = max(0, self.stats["current_visitors"] - 1)
        
        # Record hourly and daily stats
        dt = datetime.fromtimestamp(timestamp)
        hour_key = dt.strftime("%Y-%m-%d %H:00")
        day_key = dt.strftime("%Y-%m-%d")
        
        self.stats["hourly_exits"][hour_key] += 1
        self.stats["daily_exits"][day_key] += 1
        
        logger.info(f"Person {person_id} exited. Current visitors: {self.stats['current_visitors']}")
    
    def get_current_footfall_count(self) -> Dict[str, int]:
        """Get the current footfall statistics"""
        return {
            "current_visitors": self.stats["current_visitors"],
            "total_entries": self.stats["total_entries"],
            "total_exits": self.stats["total_exits"]
        }
    
    def get_hourly_stats(self, date: str = None) -> Dict[str, Dict[str, int]]:
        """
        Get hourly footfall statistics for a specific date.
        
        Args:
            date: Date string in format 'YYYY-MM-DD', defaults to today
            
        Returns:
            Dictionary with hourly entries and exits
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        hourly_entries = {}
        hourly_exits = {}
        
        for hour_key, count in self.stats["hourly_entries"].items():
            if hour_key.startswith(date):
                hour = hour_key.split()[1][:2]  # Extract hour
                hourly_entries[hour] = count
        
        for hour_key, count in self.stats["hourly_exits"].items():
            if hour_key.startswith(date):
                hour = hour_key.split()[1][:2]  # Extract hour
                hourly_exits[hour] = count
        
        return {
            "entries": hourly_entries,
            "exits": hourly_exits
        }
    
    def get_daily_stats(self, month: str = None) -> Dict[str, Dict[str, int]]:
        """
        Get daily footfall statistics for a specific month.
        
        Args:
            month: Month string in format 'YYYY-MM', defaults to current month
            
        Returns:
            Dictionary with daily entries and exits
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        daily_entries = {}
        daily_exits = {}
        
        for day_key, count in self.stats["daily_entries"].items():
            if day_key.startswith(month):
                day = day_key.split("-")[2]  # Extract day
                daily_entries[day] = count
        
        for day_key, count in self.stats["daily_exits"].items():
            if day_key.startswith(month):
                day = day_key.split("-")[2]  # Extract day
                daily_exits[day] = count
        
        return {
            "entries": daily_entries,
            "exits": daily_exits
        }
    
    def _is_in_polygon(self, point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
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
    
    def _save_stats(self):
        """Save statistics to disk"""
        stats_file = os.path.join(self.data_dir, "footfall_stats.json")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats_to_save = {
            "total_entries": self.stats["total_entries"],
            "total_exits": self.stats["total_exits"],
            "current_visitors": self.stats["current_visitors"],
            "hourly_entries": dict(self.stats["hourly_entries"]),
            "hourly_exits": dict(self.stats["hourly_exits"]),
            "daily_entries": dict(self.stats["daily_entries"]),
            "daily_exits": dict(self.stats["daily_exits"]),
            "last_updated": time.time()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        logger.info(f"Footfall statistics saved to {stats_file}")
    
    def _load_stats(self):
        """Load statistics from disk if available"""
        stats_file = os.path.join(self.data_dir, "footfall_stats.json")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    saved_stats = json.load(f)
                
                self.stats["total_entries"] = saved_stats["total_entries"]
                self.stats["total_exits"] = saved_stats["total_exits"]
                self.stats["current_visitors"] = saved_stats["current_visitors"]
                
                # Convert back to defaultdicts
                for hour_key, count in saved_stats["hourly_entries"].items():
                    self.stats["hourly_entries"][hour_key] = count
                
                for hour_key, count in saved_stats["hourly_exits"].items():
                    self.stats["hourly_exits"][hour_key] = count
                
                for day_key, count in saved_stats["daily_entries"].items():
                    self.stats["daily_entries"][day_key] = count
                
                for day_key, count in saved_stats["daily_exits"].items():
                    self.stats["daily_exits"][day_key] = count
                
                logger.info(f"Loaded footfall statistics from {stats_file}")
            except Exception as e:
                logger.error(f"Failed to load footfall statistics: {e}")
    
    def cleanup_stale_objects(self, max_age: float = 30.0):
        """Remove people that haven't been seen for a while"""
        current_time = time.time()
        to_remove = []
        
        for person_id, person_data in self.tracked_people.items():
            if current_time - person_data["last_seen"] > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_people[person_id]
            del self.zone_presence[person_id]
    
    def get_person_data(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get tracking data for a specific person.
        
        Args:
            person_id: The ID of the person to retrieve data for
            
        Returns:
            Dictionary with person tracking data or None if person not found
        """
        return self.tracked_people.get(person_id)
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the defined zones on the frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Frame with zones drawn on it
        """
        h, w = frame.shape[:2]
        
        # Make a copy of the frame to avoid modifying the original
        result = frame.copy()
        
        # Draw each zone with a different color
        colors = {
            self.entry_roi_id: (0, 255, 0),    # Green for entry
            self.exit_roi_id: (0, 0, 255),     # Red for exit
        }
        
        for zone_id, zone_poly in self.roi_zones.items():
            # Convert normalized coordinates to pixel coordinates
            points = []
            for x, y in zone_poly:
                points.append((int(x * w), int(y * h)))
            
            # Get color, default to yellow if not entry or exit
            color = colors.get(zone_id, (0, 255, 255))
            
            # Draw the polygon
            cv2.polylines(result, [np.array(points)], True, color, 2)
            
            # Add zone label
            centroid_x = sum(p[0] for p in points) // len(points)
            centroid_y = sum(p[1] for p in points) // len(points)
            
            if zone_id == self.entry_roi_id:
                label = "Entry"
            elif zone_id == self.exit_roi_id:
                label = "Exit"
            else:
                label = f"Zone {zone_id}"
            
            cv2.putText(result, label, (centroid_x, centroid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result 
