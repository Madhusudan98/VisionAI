"""
Visitor Counter

A specialized module that counts entries and exits of visitors.
This module has a single responsibility: maintain accurate counts of people entering and exiting.
"""

import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisitorCounter:
    """
    Specialized counter for tracking visitor entries and exits.
    Only responsible for maintaining accurate counts and statistics.
    """
    
    def __init__(
        self,
        data_dir: str = "data/footfall",
        store_id: str = "default_store",
        save_interval: int = 300,  # Save data every 5 minutes
        callback: Optional[Callable] = None
    ):
        """
        Initialize the visitor counter.
        
        Args:
            data_dir: Directory to save counter data
            store_id: Unique identifier for this store/location
            save_interval: How often to save count data (in seconds)
            callback: Optional callback function called when counts change
        """
        self.data_dir = data_dir
        self.store_id = store_id
        self.save_interval = save_interval
        self.callback = callback
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize counters
        self.total_entries = 0
        self.total_exits = 0
        self.current_visitors = 0
        
        # Track hourly statistics
        self.hourly_stats = {}  # {hour_key: {"entries": count, "exits": count}}
        
        # Track last save time
        self.last_save_time = time.time()
        
        # Load previous data if available
        self._load_data()
        
        logger.info(f"Visitor counter initialized for store {store_id}")
    
    def record_entry(self, timestamp: float = None) -> Dict[str, Any]:
        """
        Record a visitor entry.
        
        Args:
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with updated counter statistics
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update counters
        self.total_entries += 1
        self.current_visitors += 1
        
        # Update hourly stats
        hour_key = self._get_hour_key(timestamp)
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {"entries": 0, "exits": 0}
        self.hourly_stats[hour_key]["entries"] += 1
        
        # Save data if interval has passed
        if timestamp - self.last_save_time >= self.save_interval:
            self._save_data()
        
        # Get current stats
        stats = self.get_current_stats()
        
        # Call callback if provided
        if self.callback:
            self.callback(stats)
        
        logger.info(f"Entry recorded. Current visitors: {self.current_visitors}")
        return stats
    
    def record_exit(self, timestamp: float = None) -> Dict[str, Any]:
        """
        Record a visitor exit.
        
        Args:
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with updated counter statistics
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update counters (ensure current_visitors doesn't go negative)
        self.total_exits += 1
        self.current_visitors = max(0, self.current_visitors - 1)
        
        # Update hourly stats
        hour_key = self._get_hour_key(timestamp)
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {"entries": 0, "exits": 0}
        self.hourly_stats[hour_key]["exits"] += 1
        
        # Save data if interval has passed
        if timestamp - self.last_save_time >= self.save_interval:
            self._save_data()
        
        # Get current stats
        stats = self.get_current_stats()
        
        # Call callback if provided
        if self.callback:
            self.callback(stats)
        
        logger.info(f"Exit recorded. Current visitors: {self.current_visitors}")
        return stats
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current visitor statistics.
        
        Returns:
            Dict with current visitor statistics
        """
        return {
            "store_id": self.store_id,
            "timestamp": time.time(),
            "current_visitors": self.current_visitors,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits
        }
    
    def get_hourly_stats(self, date_str: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Get hourly statistics for a specific date.
        
        Args:
            date_str: Date string in format "YYYY-MM-DD", defaults to today
            
        Returns:
            Dict with hourly statistics
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Filter hourly stats for the specified date
        result = {}
        for hour_key, stats in self.hourly_stats.items():
            if hour_key.startswith(date_str):
                result[hour_key] = stats
        
        return result
    
    def reset_daily_counts(self):
        """Reset daily counts while maintaining historical data."""
        # Save current data first
        self._save_data()
        
        # Reset current visitors (but keep total counts)
        self.current_visitors = 0
        
        logger.info("Daily visitor counts reset")
    
    def _get_hour_key(self, timestamp: float) -> str:
        """
        Get hour key for statistics in format "YYYY-MM-DD HH".
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Hour key string
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H")
    
    def _save_data(self):
        """Save counter data to file."""
        data = {
            "store_id": self.store_id,
            "timestamp": time.time(),
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "current_visitors": self.current_visitors,
            "hourly_stats": self.hourly_stats
        }
        
        # Create filename with store ID
        filename = os.path.join(self.data_dir, f"visitor_counts_{self.store_id}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save_time = time.time()
            logger.info(f"Visitor count data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving visitor count data: {e}")
    
    def _load_data(self):
        """Load counter data from file if available."""
        filename = os.path.join(self.data_dir, f"visitor_counts_{self.store_id}.json")
        
        if not os.path.exists(filename):
            logger.info(f"No previous visitor count data found at {filename}")
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.total_entries = data.get("total_entries", 0)
            self.total_exits = data.get("total_exits", 0)
            self.current_visitors = data.get("current_visitors", 0)
            self.hourly_stats = data.get("hourly_stats", {})
            
            logger.info(f"Loaded visitor count data from {filename}")
        except Exception as e:
            logger.error(f"Error loading visitor count data: {e}")
            # Initialize with empty data on error
            self.total_entries = 0
            self.total_exits = 0
            self.current_visitors = 0
            self.hourly_stats = {} 