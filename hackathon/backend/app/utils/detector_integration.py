"""
Detector Integration

A module that demonstrates how to integrate the specialized detector components.
This shows how to combine the single-purpose modules to create a complete system.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import os

# Import specialized components
from .line_crossing_detector import LineCrossingDetector
from .visitor_counter import VisitorCounter
from .zone_detector import ZoneDetector
from .dwell_time_analyzer import DwellTimeAnalyzer
from .movement_anomaly_detector import MovementAnomalyDetector
from .unauthorized_area_detector import UnauthorizedAreaDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectorIntegration:
    """
    Integration class that shows how to use the specialized detector components together.
    """
    
    def __init__(
        self,
        store_id: str = "default_store",
        data_dir: str = "data",
        labels_file: Optional[str] = None,
        callback: Optional[Callable] = None
    ):
        """
        Initialize the detector integration.
        
        Args:
            store_id: Unique identifier for this store/location
            data_dir: Base directory for data storage
            labels_file: Path to CSV file containing label data for unauthorized areas
            callback: Optional callback function for all alerts
        """
        self.store_id = store_id
        self.data_dir = data_dir
        self.callback = callback
        
        # Ensure data directories exist
        os.makedirs(os.path.join(data_dir, "footfall"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "dwell"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "anomalies"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "unauthorized"), exist_ok=True)
        
        # Define entry/exit lines
        self.entry_line = LineCrossingDetector(
            line_start=(0.0, 0.5),
            line_end=(0.2, 0.5),
            line_id="entry_line",
            crossing_direction="positive",  # Only count entries
            callback=self._on_line_crossed
        )
        
        self.exit_line = LineCrossingDetector(
            line_start=(0.8, 0.5),
            line_end=(1.0, 0.5),
            line_id="exit_line",
            crossing_direction="negative",  # Only count exits
            callback=self._on_line_crossed
        )
        
        # Define visitor counter
        self.visitor_counter = VisitorCounter(
            store_id=store_id,
            data_dir=os.path.join(data_dir, "footfall"),
            callback=self._on_visitor_count_changed
        )
        
        # Define zones
        self.zones = {
            "cash_counter": np.array([
                [0.4, 0.3],  # top-left
                [0.6, 0.3],  # top-right
                [0.6, 0.4],  # bottom-right
                [0.4, 0.4]   # bottom-left
            ]),
            "entrance": np.array([
                [0.0, 0.4],
                [0.2, 0.4],
                [0.2, 0.6],
                [0.0, 0.6]
            ]),
            "exit": np.array([
                [0.8, 0.4],
                [1.0, 0.4],
                [1.0, 0.6],
                [0.8, 0.6]
            ])
        }
        
        # Define zone detector
        self.zone_detector = ZoneDetector(
            zones=self.zones,
            callback=self._on_zone_transition
        )
        
        # Define dwell time analyzer for cash counter
        self.dwell_analyzer = DwellTimeAnalyzer(
            areas={"cash_counter": self.zones["cash_counter"]},
            thresholds={"cash_counter": {"short": 3.0, "normal": 30.0, "long": 120.0}},
            data_dir=os.path.join(data_dir, "dwell"),
            callback=self._on_dwell_alert
        )
        
        # Define movement anomaly detector
        self.anomaly_detector = MovementAnomalyDetector(
            data_dir=os.path.join(data_dir, "anomalies"),
            callback=self._on_anomaly_alert
        )
        
        # Define unauthorized area detector
        self.unauthorized_detector = UnauthorizedAreaDetector(
            labels_file=labels_file,
            data_dir=os.path.join(data_dir, "unauthorized"),
            callback=self._on_unauthorized_access
        )
        
        logger.info(f"Detector integration initialized for store {store_id}")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> List[Dict]:
        """
        Update object position and check all detectors.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            List of alerts/events from all detectors
        """
        if timestamp is None:
            timestamp = time.time()
        
        alerts = []
        
        # Check entry/exit lines
        entry_event = self.entry_line.update(object_id, position, timestamp)
        if entry_event:
            alerts.append(entry_event)
        
        exit_event = self.exit_line.update(object_id, position, timestamp)
        if exit_event:
            alerts.append(exit_event)
        
        # Check zones
        zone_event = self.zone_detector.update(object_id, position, timestamp)
        if zone_event:
            alerts.append(zone_event)
        
        # Check dwell times
        dwell_alert = self.dwell_analyzer.update(object_id, position, timestamp)
        if dwell_alert:
            alerts.append(dwell_alert)
        
        # Check movement patterns
        anomaly_alert = self.anomaly_detector.update(object_id, position, timestamp)
        if anomaly_alert:
            alerts.append(anomaly_alert)
        
        # Check unauthorized areas
        unauthorized_alert = self.unauthorized_detector.update(object_id, position, timestamp)
        if unauthorized_alert:
            alerts.append(unauthorized_alert)
        
        return alerts
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current statistics from all detectors.
        
        Returns:
            Dictionary with combined statistics
        """
        stats = {
            "store_id": self.store_id,
            "timestamp": time.time(),
            "visitor_stats": self.visitor_counter.get_current_stats(),
            "zone_occupancy": {
                zone_id: len(self.zone_detector.get_objects_in_zone(zone_id))
                for zone_id in self.zones
            },
            "unauthorized_areas": {
                "total_events": len(self.unauthorized_detector.access_events),
                "objects_in_areas": self.unauthorized_detector.get_objects_in_unauthorized_areas()
            },
            "recent_alerts": self._get_recent_alerts()
        }
        
        return stats
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw visualization from all detectors on a frame.
        
        Args:
            frame: The frame to draw on
            
        Returns:
            Frame with visualization elements drawn
        """
        # Draw entry/exit lines
        frame = self.entry_line.draw_line(frame, color=(0, 255, 0))  # Green for entry
        frame = self.exit_line.draw_line(frame, color=(0, 0, 255))   # Red for exit
        
        # Draw zones
        frame = self.zone_detector.draw_zones(frame)
        
        # Draw dwell time areas
        frame = self.dwell_analyzer.draw_areas(frame)
        
        # Draw unauthorized areas
        frame = self.unauthorized_detector.draw_areas(frame, color=(255, 0, 0), alpha=0.4)
        
        # Add visitor stats overlay
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        
        stats = self.visitor_counter.get_current_stats()
        
        # Draw transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw stats
        cv2.putText(frame, f"Current Visitors: {stats['current_visitors']}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Entries: {stats['total_entries']}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Exits: {stats['total_exits']}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _on_line_crossed(self, event: Dict[str, Any]):
        """Handle line crossing events."""
        if event["line_id"] == "entry_line":
            self.visitor_counter.record_entry(event["timestamp"])
            logger.info(f"Entry recorded for object {event['object_id']}")
        
        elif event["line_id"] == "exit_line":
            self.visitor_counter.record_exit(event["timestamp"])
            logger.info(f"Exit recorded for object {event['object_id']}")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback(event)
    
    def _on_visitor_count_changed(self, stats: Dict[str, Any]):
        """Handle visitor count changes."""
        logger.info(f"Visitor count updated: {stats['current_visitors']} currently in store")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback({"type": "visitor_count_changed", "stats": stats})
    
    def _on_zone_transition(self, event: Dict[str, Any]):
        """Handle zone transition events."""
        logger.info(f"Object {event['object_id']} {event['transition_type']}ed zone {event['zone_id']}")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback(event)
    
    def _on_dwell_alert(self, alert: Dict[str, Any]):
        """Handle dwell time alerts."""
        logger.info(f"Dwell time alert: {alert['message']}")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback(alert)
    
    def _on_anomaly_alert(self, alert: Dict[str, Any]):
        """Handle movement anomaly alerts."""
        logger.info(f"Movement anomaly alert: {alert['message']}")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback(alert)
    
    def _on_unauthorized_access(self, alert: Dict[str, Any]):
        """Handle unauthorized area access alerts."""
        logger.warning(f"UNAUTHORIZED ACCESS ALERT: {alert['message']}")
        
        # Forward to main callback if provided
        if self.callback:
            self.callback(alert)
    
    def _get_recent_alerts(self, count: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent alerts from all detectors."""
        return {
            "dwell_alerts": self.dwell_analyzer.get_recent_alerts(count),
            "anomaly_alerts": self.anomaly_detector.get_recent_alerts(count),
            "zone_events": self.zone_detector.get_recent_events(count),
            "entry_events": self.entry_line.get_recent_crossings(count),
            "exit_events": self.exit_line.get_recent_crossings(count),
            "unauthorized_events": self.unauthorized_detector.get_recent_events(count)
        } 