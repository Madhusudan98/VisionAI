import cv2
import numpy as np
import time
import logging
import os
import json
from datetime import datetime
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TheftDetector")

class TheftDetector:
    """
    A class to detect potential theft incidents at a cash counter by analyzing movement patterns,
    dwell time, quick grabs, and unauthorized zone entries.
    """
    
    def __init__(self, cash_counter_roi, dwell_time_threshold=10.0, quick_grab_threshold=1.5, 
                 suspicious_movement_threshold=0.15, detection_cooldown=10.0, max_alerts=100):
        """
        Initialize the theft detector with ROI and thresholds.
        
        Args:
            cash_counter_roi (np.array): Region of interest for cash counter area (normalized coordinates)
            dwell_time_threshold (float): Time in seconds that's considered suspicious for dwelling at counter
            quick_grab_threshold (float): Time threshold for quick grab actions (seconds)
            suspicious_movement_threshold (float): Threshold for movement that's considered suspicious
            detection_cooldown (float): Cooldown period between alerts for the same person (seconds)
            max_alerts (int): Maximum number of alerts to store
        """
        self.cash_counter_roi = cash_counter_roi
        self.dwell_time_threshold = dwell_time_threshold
        self.quick_grab_threshold = quick_grab_threshold
        self.suspicious_movement_threshold = suspicious_movement_threshold
        self.detection_cooldown = detection_cooldown
        self.max_alerts = max_alerts
        
        # Define zones based on cash_counter_roi
        self._define_zones()
        
        # Person tracking data
        self.persons = {}  # {tracking_id: {position, history, alerts, etc.}}
        
        # Object tracking data
        self.objects = {}  # {tracking_id: {class, position, history, etc.}}
        
        # Alert history
        self.alerts = []  # List of all alerts generated
        
        # Last alert timestamps per person (for cooldown)
        self.last_alert_time = defaultdict(float)
        
        # Capture events with timestamps
        self.events = []
        
        logger.info("TheftDetector initialized")
    
    def _define_zones(self):
        """Define customer and cashier zones based on the cash counter ROI"""
        # Extract coordinates of cash counter
        min_x = min(p[0] for p in self.cash_counter_roi)
        max_x = max(p[0] for p in self.cash_counter_roi)
        min_y = min(p[1] for p in self.cash_counter_roi)
        max_y = max(p[1] for p in self.cash_counter_roi)
        
        # Cashier zone is behind the counter (top half)
        self.cashier_zone = np.array([
            [min_x, min_y],  # top-left
            [max_x, min_y],  # top-right
            [max_x, (min_y + max_y) / 2],  # middle-right
            [min_x, (min_y + max_y) / 2]   # middle-left
        ])
        
        # Customer zone is in front of the counter (bottom half)
        self.customer_zone = np.array([
            [min_x, (min_y + max_y) / 2],  # middle-left
            [max_x, (min_y + max_y) / 2],  # middle-right
            [max_x, max_y],  # bottom-right
            [min_x, max_y]   # bottom-left
        ])
        
        # Define approach zones (area in front of the counter)
        approach_depth = 0.15  # 15% of frame height for approach zone
        
        self.approach_zone = np.array([
            [min_x, max_y],  # top-left (bottom of customer zone)
            [max_x, max_y],  # top-right (bottom of customer zone)
            [max_x, max_y + approach_depth],  # bottom-right
            [min_x, max_y + approach_depth]   # bottom-left
        ])
    
    def update_person_position(self, tracking_id, position, size=None, timestamp=None):
        """
        Update a person's position and check for suspicious activities.
        
        Args:
            tracking_id (int): Unique identifier for the person
            position (tuple): (x, y) normalized coordinates
            size (float, optional): Size of the person bounding box (area)
            timestamp (float, optional): Current timestamp, defaults to time.time()
            
        Returns:
            dict or None: Alert dict if a theft is detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        x, y = position
        
        # Initialize person data if new
        if tracking_id not in self.persons:
            self.persons[tracking_id] = {
                'position': position,
                'last_position': position,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'history': deque(maxlen=50),  # Store last 50 positions
                'zone_history': deque(maxlen=10),  # Store last 10 zones
                'in_cash_counter': self._is_in_cash_counter(position),
                'in_cashier_zone': self._is_in_cashier_zone(position),
                'counter_entry_time': None,
                'size': size or 0.0,
                'movement': [],  # Movement vectors
                'alert_cooldown': 0.0  # Cooldown timer for alerts
            }
            
            # Log new person
            logger.debug(f"New person {tracking_id} detected at {position}")
        
        # Get person data
        person = self.persons[tracking_id]
        
        # Update basic information
        person['last_position'] = person['position']
        person['position'] = position
        person['last_seen'] = timestamp
        if size is not None:
            person['size'] = size
        
        # Add to position history
        person['history'].append((position, timestamp))
        
        # Calculate movement vector
        if len(person['history']) >= 2:
            prev_pos, prev_time = person['history'][-2]
            curr_pos = position
            
            if prev_time != timestamp:  # Avoid division by zero
                # Calculate movement vector and speed
                movement_x = curr_pos[0] - prev_pos[0]
                movement_y = curr_pos[1] - prev_pos[1]
                movement_vector = (movement_x, movement_y)
                
                # Calculate movement speed (distance per second)
                time_diff = timestamp - prev_time
                distance = np.sqrt(movement_x**2 + movement_y**2)
                speed = distance / time_diff if time_diff > 0 else 0
                
                # Store movement data
                person['movement'].append({
                    'vector': movement_vector,
                    'speed': speed,
                    'timestamp': timestamp
                })
                
                # Keep only last 10 movements
                if len(person['movement']) > 10:
                    person['movement'].pop(0)
        
        # Check current zone and update zone history
        was_in_cash_counter = person['in_cash_counter']
        was_in_cashier_zone = person['in_cashier_zone']
        
        # Update current zone flags
        person['in_cash_counter'] = self._is_in_cash_counter(position)
        person['in_cashier_zone'] = self._is_in_cashier_zone(position)
        
        # If person entered cash counter area, record time
        if person['in_cash_counter'] and not was_in_cash_counter:
            person['counter_entry_time'] = timestamp
            logger.debug(f"Person {tracking_id} entered cash counter area")
            
            # Add to zone history
            person['zone_history'].append(('cash_counter_entry', timestamp))
        
        # If person exited cash counter area
        elif was_in_cash_counter and not person['in_cash_counter']:
            logger.debug(f"Person {tracking_id} exited cash counter area")
            
            # Add to zone history
            person['zone_history'].append(('cash_counter_exit', timestamp))
            
            # Check if this was a quick visit (potential grab and go)
            if person['counter_entry_time'] is not None:
                dwell_time = timestamp - person['counter_entry_time']
                if dwell_time < self.quick_grab_threshold:
                    # This may be a quick grab, check if we should alert
                    if timestamp - self.last_alert_time[tracking_id] > self.detection_cooldown:
                        alert = self._generate_theft_alert(
                            tracking_id, 
                            "quick_grab", 
                            f"Quick grab detected: Person {tracking_id} was at counter for only {dwell_time:.1f}s",
                            timestamp
                        )
                        self.last_alert_time[tracking_id] = timestamp
                        return alert
        
        # If person entered cashier zone (unauthorized area)
        if person['in_cashier_zone'] and not was_in_cashier_zone:
            logger.debug(f"Person {tracking_id} entered cashier zone")
            
            # Add to zone history
            person['zone_history'].append(('cashier_zone_entry', timestamp))
            
            # Generate unauthorized entry alert
            if timestamp - self.last_alert_time[tracking_id] > self.detection_cooldown:
                alert = self._generate_theft_alert(
                    tracking_id, 
                    "unauthorized_entry", 
                    f"Unauthorized entry: Person {tracking_id} entered cashier zone",
                    timestamp
                )
                self.last_alert_time[tracking_id] = timestamp
                return alert
        
        # Check for suspicious dwell time at counter
        if person['in_cash_counter'] and person['counter_entry_time'] is not None:
            dwell_time = timestamp - person['counter_entry_time']
            if dwell_time > self.dwell_time_threshold:
                # This may be suspicious loitering, check if we should alert
                if timestamp - self.last_alert_time[tracking_id] > self.detection_cooldown:
                    alert = self._generate_theft_alert(
                        tracking_id, 
                        "suspicious_dwell_time", 
                        f"Suspicious dwell time: Person {tracking_id} at counter for {dwell_time:.1f}s",
                        timestamp
                    )
                    self.last_alert_time[tracking_id] = timestamp
                    return alert
        
        # Check for suspicious movement patterns (rapid movements, unusual patterns)
        if len(person['movement']) >= 3:
            # Calculate average speed over last few movements
            recent_speeds = [m['speed'] for m in person['movement'][-3:]]
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            
            # Check for suspicious movement patterns
            if avg_speed > self.suspicious_movement_threshold and person['in_cash_counter']:
                # Suspicious rapid movement near cash counter
                if timestamp - self.last_alert_time[tracking_id] > self.detection_cooldown:
                    alert = self._generate_theft_alert(
                        tracking_id, 
                        "suspicious_movement", 
                        f"Suspicious movement: Person {tracking_id} moving rapidly near counter",
                        timestamp
                    )
                    self.last_alert_time[tracking_id] = timestamp
                    return alert
        
        # No alerts triggered
        return None
    
    def update_object_position(self, tracking_id, object_class, position, timestamp=None):
        """
        Update position of an object (like phone, wallet, etc.) and check if it's being taken.
        
        Args:
            tracking_id (int): Unique identifier for the object
            object_class (str): Class name of the object
            position (tuple): (x, y) normalized coordinates
            timestamp (float, optional): Current timestamp, defaults to time.time()
            
        Returns:
            dict or None: Alert dict if a theft is detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        x, y = position
        
        # Initialize object data if new
        if tracking_id not in self.objects:
            self.objects[tracking_id] = {
                'class': object_class,
                'position': position,
                'last_position': position,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'history': deque(maxlen=20),  # Store last 20 positions
                'in_cash_counter': self._is_in_cash_counter(position),
                'in_cashier_zone': self._is_in_cashier_zone(position),
                'taken_from_counter': False
            }
            
            # Log new object
            logger.debug(f"New object {tracking_id} ({object_class}) detected at {position}")
        
        # Get object data
        obj = self.objects[tracking_id]
        
        # Update object class if it changed
        if obj['class'] != object_class:
            obj['class'] = object_class
        
        # Update basic information
        obj['last_position'] = obj['position']
        obj['position'] = position
        obj['last_seen'] = timestamp
        
        # Add to position history
        obj['history'].append((position, timestamp))
        
        # Check current zone
        was_in_cash_counter = obj['in_cash_counter']
        was_in_cashier_zone = obj['in_cashier_zone']
        
        # Update current zone flags
        obj['in_cash_counter'] = self._is_in_cash_counter(position)
        obj['in_cashier_zone'] = self._is_in_cashier_zone(position)
        
        # If object was on counter and now isn't, mark as potentially taken
        if was_in_cash_counter and not obj['in_cash_counter']:
            obj['taken_from_counter'] = True
            
            # Find nearest person
            nearest_person_id = self._find_nearest_person(position, timestamp)
            
            if nearest_person_id is not None:
                # Check if we should raise an alert
                person = self.persons[nearest_person_id]
                
                # If the person is not the cashier (not in cashier zone)
                if not person['in_cashier_zone']:
                    # This might be a theft, check cooldown
                    if timestamp - self.last_alert_time[nearest_person_id] > self.detection_cooldown:
                        alert = self._generate_theft_alert(
                            nearest_person_id, 
                            "object_taken", 
                            f"Object taken: {object_class} possibly taken by person {nearest_person_id}",
                            timestamp,
                            object_data={
                                'object_id': tracking_id,
                                'object_class': object_class
                            }
                        )
                        self.last_alert_time[nearest_person_id] = timestamp
                        return alert
        
        # No alerts triggered
        return None
    
    def _is_in_cash_counter(self, position):
        """Check if a position is within the cash counter ROI"""
        x, y = position
        roi = self.cash_counter_roi
        
        # Convert to numpy array for pointPolygonTest
        point = np.array([x, y])
        polygon = np.array(roi)
        
        # Check if point is inside polygon using point-in-polygon test
        result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
        return result >= 0
    
    def _is_in_cashier_zone(self, position):
        """Check if a position is within the cashier zone"""
        x, y = position
        roi = self.cashier_zone
        
        # Convert to numpy array for pointPolygonTest
        point = np.array([x, y])
        polygon = np.array(roi)
        
        # Check if point is inside polygon using point-in-polygon test
        result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
        return result >= 0
    
    def _find_nearest_person(self, position, timestamp, max_age=1.0, max_distance=0.2):
        """
        Find the nearest person to a given position.
        
        Args:
            position (tuple): Position to check against (x, y)
            timestamp (float): Current timestamp
            max_age (float): Max age in seconds to consider a person active
            max_distance (float): Max distance to consider
            
        Returns:
            int or None: Tracking ID of the nearest person, or None if none found
        """
        nearest_id = None
        min_distance = float('inf')
        
        for person_id, person in self.persons.items():
            # Skip if person hasn't been seen recently
            if timestamp - person['last_seen'] > max_age:
                continue
            
            # Calculate distance
            person_pos = person['position']
            distance = np.sqrt((position[0] - person_pos[0])**2 + (position[1] - person_pos[1])**2)
            
            # Check if this is the nearest so far and within max distance
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_id = person_id
        
        return nearest_id
    
    def _generate_theft_alert(self, person_id, alert_type, message, timestamp, object_data=None):
        """
        Generate a theft alert and add it to the alerts list.
        
        Args:
            person_id (int): ID of the person involved
            alert_type (str): Type of alert (e.g., "quick_grab", "unauthorized_entry")
            message (str): Alert message
            timestamp (float): Alert timestamp
            object_data (dict, optional): Additional data about the object involved
            
        Returns:
            dict: The generated alert
        """
        # Create alert
        alert = {
            'id': len(self.alerts) + 1,
            'person_id': person_id,
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'position': self.persons[person_id]['position'] if person_id in self.persons else None,
            'object_data': object_data
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Ensure we don't exceed max alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        # Log the alert
        logger.warning(f"THEFT ALERT: {message}")
        
        return alert
    
    def cleanup_stale_objects(self, max_age=5.0):
        """
        Remove stale persons and objects from tracking.
        
        Args:
            max_age (float): Maximum age in seconds before considering an object/person stale
        """
        current_time = time.time()
        
        # Clean up stale persons
        stale_persons = []
        for person_id, person in self.persons.items():
            if current_time - person['last_seen'] > max_age:
                stale_persons.append(person_id)
        
        for person_id in stale_persons:
            self.persons.pop(person_id)
        
        # Clean up stale objects
        stale_objects = []
        for object_id, obj in self.objects.items():
            if current_time - obj['last_seen'] > max_age:
                stale_objects.append(object_id)
        
        for object_id in stale_objects:
            self.objects.pop(object_id)
    
    def get_alerts(self, max_age=None, limit=None):
        """
        Get list of alerts, optionally filtered by recency.
        
        Args:
            max_age (float, optional): Maximum age in seconds for alerts to include
            limit (int, optional): Maximum number of alerts to return
            
        Returns:
            list: List of alert dictionaries
        """
        if max_age is None and limit is None:
            # Return all alerts
            return self.alerts
        
        # Filter by age if needed
        filtered_alerts = self.alerts
        if max_age is not None:
            current_time = time.time()
            filtered_alerts = [
                alert for alert in self.alerts
                if current_time - alert['timestamp'] <= max_age
            ]
        
        # Apply limit if needed
        if limit is not None:
            filtered_alerts = filtered_alerts[-limit:]
        
        return filtered_alerts
    
    def draw_zones(self, frame, frame_width=None, frame_height=None):
        """
        Draw ROI zones on a frame for visualization.
        
        Args:
            frame: OpenCV frame
            frame_width (int, optional): Frame width, defaults to frame.shape[1]
            frame_height (int, optional): Frame height, defaults to frame.shape[0]
            
        Returns:
            frame: Frame with zones drawn
        """
        if frame_width is None:
            frame_width = frame.shape[1]
        if frame_height is None:
            frame_height = frame.shape[0]
        
        # Draw cash counter ROI
        counter_points = np.array([
            [int(p[0] * frame_width), int(p[1] * frame_height)]
            for p in self.cash_counter_roi
        ])
        cv2.polylines(frame, [counter_points], True, (0, 255, 255), 2)
        cv2.putText(frame, "Cash Counter", 
                   (int(counter_points[0][0]), int(counter_points[0][1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw cashier zone
        cashier_points = np.array([
            [int(p[0] * frame_width), int(p[1] * frame_height)]
            for p in self.cashier_zone
        ])
        cv2.polylines(frame, [cashier_points], True, (255, 0, 255), 2)
        cv2.putText(frame, "Cashier Zone", 
                   (int(cashier_points[0][0]), int(cashier_points[0][1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw customer zone
        customer_points = np.array([
            [int(p[0] * frame_width), int(p[1] * frame_height)]
            for p in self.customer_zone
        ])
        cv2.polylines(frame, [customer_points], True, (0, 255, 0), 2)
        cv2.putText(frame, "Customer Zone", 
                   (int(customer_points[0][0]), int(customer_points[0][1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def display_alert_overlay(self, frame, alert):
        """
        Display an alert overlay on the frame.
        
        Args:
            frame: OpenCV frame
            alert (dict): Alert data
            
        Returns:
            frame: Frame with alert overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw red border around frame
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 0, 255), 10)
        
        # Draw alert box
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add alert text
        cv2.putText(frame, "THEFT ALERT", (w//4 + 20, h//4 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.putText(frame, f"Type: {alert['type']}", (w//4 + 20, h//4 + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Person ID: {alert['person_id']}", (w//4 + 20, h//4 + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Split message into multiple lines if too long
        message = alert['message']
        y_pos = h//4 + 180
        while len(message) > 0:
            line = message[:40]  # 40 chars per line
            message = message[40:]
            cv2.putText(frame, line, (w//4 + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # Time
        cv2.putText(frame, f"Time: {alert['datetime']}", (w//4 + 20, 3*h//4 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def get_status(self):
        """Get the current status of the theft detector"""
        return {
            'persons_tracked': len(self.persons),
            'objects_tracked': len(self.objects),
            'alert_count': len(self.alerts),
            'recent_alerts': self.get_alerts(max_age=60, limit=5),  # Last minute, max 5 alerts
            'config': {
                'dwell_time_threshold': self.dwell_time_threshold,
                'quick_grab_threshold': self.quick_grab_threshold,
                'suspicious_movement_threshold': self.suspicious_movement_threshold
            }
        }
    
    def reset(self):
        """Reset all tracking data"""
        self.persons = {}
        self.objects = {}
        self.alerts = []
        self.last_alert_time = defaultdict(float)
        self.events = []
        logger.info("TheftDetector reset") 