import cv2
import numpy as np
import os
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FootfallTracker")

class FootfallTracker:
    """
    Tracks visitor footfall in a retail environment, including entries, exits, 
    and zone-based analytics.
    """
    
    def __init__(self, roi_zones, entry_roi_id=None, exit_roi_id=None, 
                 data_dir='data/footfall', camera_id=None):
        """
        Initialize the footfall tracker.
        
        Args:
            roi_zones (dict): Dictionary mapping zone IDs to zone polygons (in normalized coordinates)
            entry_roi_id (int): ID of the entry zone
            exit_roi_id (int): ID of the exit zone
            data_dir (str): Directory to save footfall data
            camera_id (int, optional): Camera ID if multiple cameras are being tracked
        """
        self.roi_zones = roi_zones
        self.entry_roi_id = entry_roi_id
        self.exit_roi_id = exit_roi_id
        self.data_dir = data_dir
        self.camera_id = camera_id
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Object tracking information
        self.tracked_objects = {}  # {tracking_id: {zone_id, last_zone, entry_time, etc.}}
        
        # Stats counters
        self.total_entries = 0
        self.total_exits = 0
        self.current_visitors = 0
        
        # Zone analytics
        self.zone_stats = {zone_id: {"entries": 0, "exits": 0, "dwell_time": []} 
                          for zone_id in roi_zones.keys()}
        
        # Heatmap data for footfall visualization
        self.heatmap_data = []
        
        # Daily and hourly statistics
        self.daily_stats = defaultdict(lambda: {"entries": 0, "exits": 0, "unique_visitors": set()})
        self.hourly_stats = defaultdict(lambda: {"entries": 0, "exits": 0, "visitors": 0})
        
        # Dwell time thresholds (seconds)
        self.short_dwell = 60  # Less than 1 minute
        self.medium_dwell = 300  # 1-5 minutes
        self.long_dwell = 600  # More than 5 minutes
        
        # Load previous data if available
        self._load_data()
        
        logger.info(f"FootfallTracker initialized with {len(roi_zones)} ROI zones")
    
    def update_object_position(self, tracking_id, position):
        """
        Update the position of a tracked object and process zone transitions.
        
        Args:
            tracking_id (int): Unique tracking ID for the object
            position (tuple): Normalized (x, y) coordinates
            
        Returns:
            dict or None: Information about zone transition if it occurred
        """
        x, y = position
        current_time = time.time()
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00:00")
        
        # Check which ROI zone the object is in
        current_zone_id = None
        for zone_id, zone in self.roi_zones.items():
            if self._is_point_in_polygon(position, zone):
                current_zone_id = zone_id
                break
        
        # Check if this is a new object
        if tracking_id not in self.tracked_objects:
            self.tracked_objects[tracking_id] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "current_zone": current_zone_id,
                "last_zone": None,
                "zone_history": [(current_zone_id, current_time)],
                "path": [position],
                "entry_time": current_time if current_zone_id == self.entry_roi_id else None,
                "exit_time": None,
                "counted_entry": False,
                "counted_exit": False
            }
            
            # Update heatmap
            self.heatmap_data.append(position)
            
            # For entry zone, just note the entry but don't increment yet
            if current_zone_id == self.entry_roi_id:
                logger.debug(f"Object {tracking_id} entered entry zone")
            
            return None
        
        # Get the existing object data
        obj_data = self.tracked_objects[tracking_id]
        
        # Update basics
        obj_data["last_seen"] = current_time
        obj_data["path"].append(position)
        
        # Skip further processing if zone hasn't changed
        if current_zone_id == obj_data["current_zone"]:
            return None
        
        # If zone has changed, process the transition
        old_zone = obj_data["current_zone"]
        obj_data["last_zone"] = old_zone
        obj_data["current_zone"] = current_zone_id
        obj_data["zone_history"].append((current_zone_id, current_time))
        
        # Handle entry counting logic
        if not obj_data["counted_entry"] and old_zone == self.entry_roi_id and current_zone_id is not None:
            # Object has fully entered from entry zone to another zone
            self.total_entries += 1
            self.current_visitors += 1
            obj_data["counted_entry"] = True
            
            # Update daily and hourly stats
            self.daily_stats[current_date]["entries"] += 1
            self.daily_stats[current_date]["unique_visitors"].add(tracking_id)
            self.hourly_stats[current_hour]["entries"] += 1
            self.hourly_stats[current_hour]["visitors"] += 1
            
            logger.info(f"Entry detected: Object {tracking_id} (Current count: {self.current_visitors})")
        
        # Handle exit counting logic
        if not obj_data["counted_exit"] and old_zone is not None and current_zone_id == self.exit_roi_id:
            # Object is exiting
            obj_data["exit_time"] = current_time
            obj_data["counted_exit"] = True
            
            if obj_data["counted_entry"]:  # Only count people who entered
                self.total_exits += 1
                self.current_visitors = max(0, self.current_visitors - 1)  # Prevent negative counts
                
                # Update daily and hourly stats
                self.daily_stats[current_date]["exits"] += 1
                self.hourly_stats[current_hour]["exits"] += 1
                
                logger.info(f"Exit detected: Object {tracking_id} (Current count: {self.current_visitors})")
        
        # Update zone-specific stats for the old zone
        if old_zone is not None:
            # Record zone exit
            self.zone_stats[old_zone]["exits"] += 1
            
            # Calculate dwell time in the zone
            zone_entry_time = None
            for zone_id, timestamp in reversed(obj_data["zone_history"][:-1]):
                if zone_id == old_zone:
                    zone_entry_time = timestamp
                    break
            
            if zone_entry_time is not None:
                dwell_time = current_time - zone_entry_time
                self.zone_stats[old_zone]["dwell_time"].append(dwell_time)
        
        # Update zone-specific stats for the new zone
        if current_zone_id is not None:
            # Record zone entry
            self.zone_stats[current_zone_id]["entries"] += 1
        
        # Periodically save the data
        if len(self.heatmap_data) % 100 == 0:
            self._save_data()
        
        return {
            "tracking_id": tracking_id,
            "from_zone": old_zone,
            "to_zone": current_zone_id,
            "timestamp": current_time
        }
    
    def cleanup_stale_objects(self, max_age=30.0):
        """
        Remove objects that haven't been seen for a while.
        
        Args:
            max_age (float): Maximum time in seconds since last update before removing an object
        """
        current_time = time.time()
        stale_ids = []
        
        for tracking_id, obj_data in self.tracked_objects.items():
            time_since_update = current_time - obj_data["last_seen"]
            
            if time_since_update > max_age:
                stale_ids.append(tracking_id)
                
                # If an object disappeared without exiting, force an exit
                if obj_data["counted_entry"] and not obj_data["counted_exit"] and self.current_visitors > 0:
                    self.total_exits += 1
                    self.current_visitors -= 1
                    
                    # Update stats
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    current_hour = datetime.now().strftime("%Y-%m-%d %H:00:00")
                    self.daily_stats[current_date]["exits"] += 1
                    self.hourly_stats[current_hour]["exits"] += 1
                    
                    logger.info(f"Forced exit for stale object {tracking_id} (Current count: {self.current_visitors})")
        
        # Remove stale objects
        for tracking_id in stale_ids:
            del self.tracked_objects[tracking_id]
    
    def get_current_footfall_count(self):
        """
        Get the current footfall count.
        
        Returns:
            dict: Current footfall statistics
        """
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "current_visitors": self.current_visitors,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_zone_analytics(self):
        """
        Get zone-specific analytics including traffic and dwell time.
        
        Returns:
            dict: Zone analytics for each zone
        """
        zone_analytics = {}
        
        for zone_id, stats in self.zone_stats.items():
            dwell_times = stats["dwell_time"]
            
            # Calculate average dwell time
            avg_dwell = sum(dwell_times) / max(len(dwell_times), 1)
            
            # Calculate dwell time distribution
            dwell_distribution = {
                "short": sum(1 for t in dwell_times if t <= self.short_dwell),
                "medium": sum(1 for t in dwell_times if self.short_dwell < t <= self.medium_dwell),
                "long": sum(1 for t in dwell_times if t > self.medium_dwell)
            }
            
            zone_analytics[zone_id] = {
                "entries": stats["entries"],
                "exits": stats["exits"],
                "traffic": stats["entries"] + stats["exits"],
                "avg_dwell_time": avg_dwell,
                "dwell_time_distribution": dwell_distribution
            }
        
        return zone_analytics
    
    def get_daily_stats(self, days=7):
        """
        Get daily statistics for the past specified number of days.
        
        Args:
            days (int): Number of days to retrieve statistics for
            
        Returns:
            dict: Daily statistics
        """
        stats = {}
        
        # Get dates for the past N days
        today = datetime.now().date()
        date_range = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        
        for date_str in date_range:
            if date_str in self.daily_stats:
                day_stats = self.daily_stats[date_str]
                stats[date_str] = {
                    "entries": day_stats["entries"],
                    "exits": day_stats["exits"],
                    "unique_visitors": len(day_stats["unique_visitors"])
                }
            else:
                stats[date_str] = {"entries": 0, "exits": 0, "unique_visitors": 0}
        
        return stats
    
    def get_hourly_stats(self, date=None):
        """
        Get hourly statistics for a specific date.
        
        Args:
            date (str, optional): Date string in format YYYY-MM-DD, defaults to today
            
        Returns:
            dict: Hourly statistics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        stats = {}
        
        # Create all hours of the day
        for hour in range(24):
            hour_key = f"{date} {hour:02d}:00:00"
            
            if hour_key in self.hourly_stats:
                hour_stats = self.hourly_stats[hour_key]
                stats[hour] = {
                    "entries": hour_stats["entries"],
                    "exits": hour_stats["exits"],
                    "visitors": hour_stats["visitors"]
                }
            else:
                stats[hour] = {"entries": 0, "exits": 0, "visitors": 0}
        
        return stats
    
    def get_heatmap_data(self):
        """
        Get footfall heatmap data.
        
        Returns:
            list: List of position points (x, y)
        """
        return self.heatmap_data
    
    def _is_point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon.
        
        Args:
            point (tuple): Point coordinates (x, y)
            polygon (ndarray): Polygon vertices
            
        Returns:
            bool: True if point is in polygon, False otherwise
        """
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
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _save_data(self):
        """Save footfall data to disk."""
        try:
            # Create a data structure to save
            data = {
                "total_entries": self.total_entries,
                "total_exits": self.total_exits,
                "current_visitors": self.current_visitors,
                "zone_stats": self.zone_stats,
                "daily_stats": {
                    date: {
                        "entries": stats["entries"],
                        "exits": stats["exits"],
                        "unique_visitors": len(stats["unique_visitors"])
                    } for date, stats in self.daily_stats.items()
                },
                "hourly_stats": dict(self.hourly_stats),
                "heatmap_data": self.heatmap_data[:1000],  # Save just a sample for heatmap
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Define the save path
            cam_suffix = f"_cam{self.camera_id}" if self.camera_id is not None else ""
            save_path = os.path.join(self.data_dir, f"footfall_data{cam_suffix}.json")
            
            # Save data to JSON file
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved footfall data to {save_path}")
        
        except Exception as e:
            logger.error(f"Error saving footfall data: {e}")
    
    def _load_data(self):
        """Load footfall data from disk."""
        try:
            # Define the load path
            cam_suffix = f"_cam{self.camera_id}" if self.camera_id is not None else ""
            load_path = os.path.join(self.data_dir, f"footfall_data{cam_suffix}.json")
            
            # Check if file exists
            if not os.path.exists(load_path):
                logger.info(f"No existing footfall data found at {load_path}")
                return
            
            # Load data from JSON file
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Restore basic counters
            self.total_entries = data.get("total_entries", 0)
            self.total_exits = data.get("total_exits", 0)
            self.current_visitors = data.get("current_visitors", 0)
            
            # Restore zone stats
            saved_zone_stats = data.get("zone_stats", {})
            for zone_id, stats in saved_zone_stats.items():
                zone_id = int(zone_id) if zone_id.isdigit() else zone_id
                if zone_id in self.zone_stats:
                    self.zone_stats[zone_id] = stats
            
            # Restore daily stats
            saved_daily_stats = data.get("daily_stats", {})
            for date, stats in saved_daily_stats.items():
                self.daily_stats[date] = {
                    "entries": stats["entries"],
                    "exits": stats["exits"],
                    "unique_visitors": set(range(stats["unique_visitors"]))  # Approximate
                }
            
            # Restore hourly stats
            saved_hourly_stats = data.get("hourly_stats", {})
            for hour, stats in saved_hourly_stats.items():
                self.hourly_stats[hour] = stats
            
            # Restore heatmap data
            self.heatmap_data = data.get("heatmap_data", [])
            
            logger.info(f"Loaded footfall data from {load_path}")
        
        except Exception as e:
            logger.error(f"Error loading footfall data: {e}")
    
    def reset_stats(self):
        """Reset all footfall statistics."""
        self.total_entries = 0
        self.total_exits = 0
        self.current_visitors = 0
        self.tracked_objects = {}
        
        # Reset zone stats
        self.zone_stats = {zone_id: {"entries": 0, "exits": 0, "dwell_time": []} 
                          for zone_id in self.roi_zones.keys()}
        
        # Reset time-based stats
        self.daily_stats = defaultdict(lambda: {"entries": 0, "exits": 0, "unique_visitors": set()})
        self.hourly_stats = defaultdict(lambda: {"entries": 0, "exits": 0, "visitors": 0})
        
        # Clear heatmap
        self.heatmap_data = []
        
        logger.info("FootfallTracker statistics reset") 