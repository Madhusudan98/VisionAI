"""
Line Crossing Detector

A specialized module that detects when a tracked object crosses a defined line or boundary.
This module has a single responsibility: detect line crossings and trigger events when they occur.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LineCrossingDetector:
    """
    Specialized detector for line/boundary crossing events.
    Only responsible for detecting when objects cross defined lines.
    """
    
    def __init__(
        self,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        line_id: str = "default_line",
        crossing_direction: str = "both",  # "both", "positive", "negative"
        cooldown_period: float = 2.0,      # seconds between repeated triggers for same ID
        callback: Optional[Callable] = None
    ):
        """
        Initialize the line crossing detector.
        
        Args:
            line_start: Starting point of line (normalized 0-1 coordinates)
            line_end: Ending point of line (normalized 0-1 coordinates)
            line_id: Unique identifier for this line
            crossing_direction: Direction that counts as crossing ("both", "positive", "negative")
            cooldown_period: Time in seconds before same object can trigger again
            callback: Optional callback function to call when line is crossed
        """
        self.line_start = line_start
        self.line_end = line_end
        self.line_id = line_id
        self.crossing_direction = crossing_direction
        self.cooldown_period = cooldown_period
        self.callback = callback
        
        # Track object positions relative to the line
        self.object_positions = {}  # {object_id: {"side": -1/1, "last_cross_time": timestamp}}
        
        # Track crossing events
        self.crossing_events = []
        
        logger.info(f"Line crossing detector initialized for line {line_id}")
    
    def update(self, object_id: int, position: Tuple[float, float], timestamp: float = None) -> Optional[Dict]:
        """
        Update object position and check if it crossed the line.
        
        Args:
            object_id: Unique ID of the object
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dict with crossing event details if line was crossed, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Determine which side of the line the object is on
        side = self._get_side_of_line(position)
        
        # If object is new, just record its position
        if object_id not in self.object_positions:
            self.object_positions[object_id] = {
                "side": side,
                "last_cross_time": 0
            }
            return None
        
        # Get previous position data
        prev_data = self.object_positions[object_id]
        prev_side = prev_data["side"]
        last_cross_time = prev_data["last_cross_time"]
        
        # Update position data
        self.object_positions[object_id]["side"] = side
        
        # Check if the object crossed the line
        if side != prev_side:
            # Check if we're still in cooldown period
            if timestamp - last_cross_time < self.cooldown_period:
                return None
            
            # Determine crossing direction
            direction = "positive" if prev_side < 0 else "negative"
            
            # Check if this crossing direction should be counted
            if self.crossing_direction != "both" and direction != self.crossing_direction:
                return None
            
            # Update last crossing time
            self.object_positions[object_id]["last_cross_time"] = timestamp
            
            # Create crossing event
            event = {
                "object_id": object_id,
                "line_id": self.line_id,
                "timestamp": timestamp,
                "direction": direction,
                "position": position
            }
            
            # Add to crossing events list
            self.crossing_events.append(event)
            
            # Call callback if provided
            if self.callback:
                self.callback(event)
            
            logger.info(f"Object {object_id} crossed line {self.line_id} in {direction} direction")
            return event
        
        return None
    
    def _get_side_of_line(self, point: Tuple[float, float]) -> int:
        """
        Determine which side of the line a point is on.
        
        Args:
            point: The point to check (x, y)
            
        Returns:
            1 if point is on one side, -1 if on the other side
        """
        x, y = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        # Calculate the side using the cross product
        value = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        # Return 1 or -1 based on which side
        return 1 if value > 0 else -1
    
    def get_recent_crossings(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent crossing events.
        
        Args:
            count: Number of recent events to return
            
        Returns:
            List of recent crossing events
        """
        return self.crossing_events[-count:] if self.crossing_events else []
    
    def reset(self):
        """Reset the detector, clearing all tracking data but keeping configuration."""
        self.object_positions = {}
        self.crossing_events = []
        logger.info(f"Line crossing detector {self.line_id} reset")
    
    def draw_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw the line on a frame.
        
        Args:
            frame: The frame to draw on
            color: RGB color tuple
            thickness: Line thickness
            
        Returns:
            Frame with line drawn
        """
        height, width = frame.shape[:2]
        start_point = (int(self.line_start[0] * width), int(self.line_start[1] * height))
        end_point = (int(self.line_end[0] * width), int(self.line_end[1] * height))
        
        import cv2  # Import here to avoid requiring cv2 for basic functionality
        cv2.line(frame, start_point, end_point, color, thickness)
        
        # Add line ID text
        text_position = (
            int((self.line_start[0] + self.line_end[0]) * width / 2),
            int((self.line_start[1] + self.line_end[1]) * height / 2) - 10
        )
        cv2.putText(frame, self.line_id, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame 