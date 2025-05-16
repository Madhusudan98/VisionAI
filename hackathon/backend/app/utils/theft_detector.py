import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TheftDetector:
    """
    Class for detecting potential theft activities in retail environments,
    particularly around cash counter areas.
    """
    
    def __init__(
        self,
        cash_counter_roi: np.ndarray,
        dwell_time_threshold: float = 30.0,
        quick_grab_threshold: float = 1.5,
        suspicious_movement_threshold: float = 0.7,
        data_dir: str = "data/theft",
        camera_id: Optional[int] = None,
        entry_zone: Optional[np.ndarray] = None,
        exit_zone: Optional[np.ndarray] = None,
        customer_zone: Optional[np.ndarray] = None,
        cashier_zone: Optional[np.ndarray] = None
    ):
        """
        Initialize the TheftDetector.
        
        Args:
            cash_counter_roi: Array of polygon coordinates defining the cash counter area (normalized 0-1)
            dwell_time_threshold: Time in seconds that's unusually long for someone to stay at counter
            quick_grab_threshold: Time in seconds that's unusually quick for an interaction
            suspicious_movement_threshold: Threshold for movement pattern to be considered suspicious
            data_dir: Directory to save theft detection data
            camera_id: Optional camera ID
            entry_zone: Optional array of polygon coordinates defining the entry zone (normalized 0-1)
            exit_zone: Optional array of polygon coordinates defining the exit zone (normalized 0-1)
            customer_zone: Optional array of polygon coordinates defining the customer zone (normalized 0-1)
            cashier_zone: Optional array of polygon coordinates defining the cashier zone (normalized 0-1)
        """
        self.cash_counter_roi = cash_counter_roi
        self.dwell_time_threshold = dwell_time_threshold
        self.quick_grab_threshold = quick_grab_threshold
        self.suspicious_movement_threshold = suspicious_movement_threshold
        self.data_dir = data_dir
        self.camera_id = camera_id
        self.entry_zone = entry_zone
        self.exit_zone = exit_zone
        
        # Track objects around cash counter
        self.tracked_objects = {}  # {object_id: data}
        
        # Track potential theft incidents
        self.theft_incidents = []
        self.confirmed_thefts = []
        self.last_alert_time = 0  # To prevent alert flooding
        self.alert_cooldown = 10.0  # Seconds between alerts
        self.suspicious_persons = {}  # {person_id: {"reason": ..., "object": ..., "timestamp": ...}}
        
        # Store positional history for movement pattern analysis
        self.position_history = defaultdict(lambda: deque(maxlen=50))
        
        # Activity zones around counter
        self.customer_zone = customer_zone  # User-defined or derived from cash_counter_roi
        self.cashier_zone = cashier_zone    # User-defined or derived from cash_counter_roi
        
        # Define customer and cashier zones based on counter ROI if not provided
        if self.customer_zone is None or self.cashier_zone is None:
            self._define_zones()
        
        logger.info("TheftDetector initialized for cash counter area")
    
    def _define_zones(self):
        """Define customer and cashier zones based on the cash counter ROI"""
        # Get min and max coordinates from the counter ROI
        min_x = min(pt[0] for pt in self.cash_counter_roi)
        max_x = max(pt[0] for pt in self.cash_counter_roi)
        min_y = min(pt[1] for pt in self.cash_counter_roi)
        max_y = max(pt[1] for pt in self.cash_counter_roi)
        
        # Define customer zone (in front of counter)
        self.customer_zone = np.array([
            [min_x - 0.05, min_y - 0.05],
            [max_x + 0.05, min_y - 0.05],
            [max_x + 0.05, min_y],
            [min_x - 0.05, min_y]
        ])
        
        # Define cashier zone (behind counter)
        self.cashier_zone = np.array([
            [min_x - 0.05, max_y],
            [max_x + 0.05, max_y],
            [max_x + 0.05, max_y + 0.05],
            [min_x - 0.05, max_y + 0.05]
        ])
    
    def update_person_position(self, person_id: int, position: Tuple[float, float], timestamp: float = None):
        """
        Update the position of a person and check for suspicious activities.
        
        Args:
            person_id: Unique ID of the person
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
        
        Returns:
            Dict with detection results if suspicious activity detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize tracking if new person
        if person_id not in self.tracked_objects:
            self.tracked_objects[person_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "positions": [(position, timestamp)],
                "in_counter_area": self._is_in_polygon(position, self.cash_counter_roi),
                "in_customer_zone": self._is_in_polygon(position, self.customer_zone),
                "in_cashier_zone": self._is_in_polygon(position, self.cashier_zone),
                "counter_entry_time": None,
                "counter_exit_time": None,
                "zone_transitions": [],
                "alert_generated": False
            }
        else:
            # Update existing person data
            person_data = self.tracked_objects[person_id]
            
            # Update position history
            person_data["positions"].append((position, timestamp))
            person_data["last_seen"] = timestamp
            
            # Check if person has entered/exited counter area
            in_counter_area = self._is_in_polygon(position, self.cash_counter_roi)
            in_customer_zone = self._is_in_polygon(position, self.customer_zone)
            in_cashier_zone = self._is_in_polygon(position, self.cashier_zone)
            
            # Track zone transitions
            if in_counter_area != person_data["in_counter_area"]:
                if in_counter_area:
                    # Person entered counter area
                    person_data["counter_entry_time"] = timestamp
                    person_data["zone_transitions"].append(("enter_counter", timestamp))
                else:
                    # Person exited counter area
                    person_data["counter_exit_time"] = timestamp
                    person_data["zone_transitions"].append(("exit_counter", timestamp))
                    
                    # Check for quick grab (unusually short time at counter)
                    if (person_data["counter_entry_time"] is not None and 
                            timestamp - person_data["counter_entry_time"] < self.quick_grab_threshold):
                        alert = self._generate_theft_alert(
                            person_id, "quick_grab", 
                            f"Person {person_id} spent only {timestamp - person_data['counter_entry_time']:.2f}s at counter",
                            timestamp
                        )
                        if alert:
                            self.suspicious_persons[person_id] = {
                                "reason": "quick_grab",
                                "object": None,
                                "timestamp": timestamp
                            }
                        return alert
            
            # Track when customer enters cashier zone (suspicious)
            if in_cashier_zone and not person_data["in_cashier_zone"] and not person_data["alert_generated"]:
                person_data["zone_transitions"].append(("enter_cashier_zone", timestamp))
                person_data["alert_generated"] = True
                alert = self._generate_theft_alert(
                    person_id, "unauthorized_zone", 
                    f"Person {person_id} entered cashier zone",
                    timestamp
                )
                if alert:
                    self.suspicious_persons[person_id] = {
                        "reason": "unauthorized_zone",
                        "object": None,
                        "timestamp": timestamp
                    }
                return alert
            
            # Track when person stays too long at counter
            if (in_counter_area and person_data["counter_entry_time"] is not None and
                    timestamp - person_data["counter_entry_time"] > self.dwell_time_threshold and
                    not person_data["alert_generated"]):
                person_data["alert_generated"] = True
                alert = self._generate_theft_alert(
                    person_id, "unusual_dwell_time", 
                    f"Person {person_id} has been at counter for {timestamp - person_data['counter_entry_time']:.2f}s",
                    timestamp
                )
                if alert:
                    self.suspicious_persons[person_id] = {
                        "reason": "unusual_dwell_time",
                        "object": None,
                        "timestamp": timestamp
                    }
                return alert
            
            # Update state flags
            person_data["in_counter_area"] = in_counter_area
            person_data["in_customer_zone"] = in_customer_zone
            person_data["in_cashier_zone"] = in_cashier_zone
        
        # Add to position history for pattern analysis
        self.position_history[person_id].append((position, timestamp))
        
        # Analyze movement patterns if we have enough history
        if len(self.position_history[person_id]) >= 10:
            suspicion_score = self._analyze_movement_pattern(person_id)
            if (suspicion_score > self.suspicious_movement_threshold and 
                    not self.tracked_objects[person_id]["alert_generated"]):
                self.tracked_objects[person_id]["alert_generated"] = True
                alert = self._generate_theft_alert(
                    person_id, "suspicious_movement", 
                    f"Person {person_id} shows suspicious movement pattern (score: {suspicion_score:.2f})",
                    timestamp
                )
                if alert:
                    self.suspicious_persons[person_id] = {
                        "reason": "suspicious_movement",
                        "object": None,
                        "timestamp": timestamp
                    }
                return alert
        
        return None  # No suspicious activity detected
    
    def update_object_position(self, object_id: int, object_class: str, position: Tuple[float, float], timestamp: float = None):
        """
        Track positions of non-person objects like bags, phones, wallets
        to detect when they might be taken from counter.
        
        Args:
            object_id: Unique ID of the object
            object_class: Class of object (bag, phone, etc.)
            position: Normalized (x, y) coordinates (0-1)
            timestamp: Optional timestamp, defaults to current time
        
        Returns:
            Dict with detection results if suspicious activity detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize tracking if new object
        obj_key = f"{object_class}_{object_id}"
        if obj_key not in self.tracked_objects:
            self.tracked_objects[obj_key] = {
                "type": "object",
                "class": object_class,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "positions": [(position, timestamp)],
                "in_counter_area": self._is_in_polygon(position, self.cash_counter_roi),
                "last_associated_person": None,
                "alert_generated": False
            }
        else:
            # Update existing object data
            object_data = self.tracked_objects[obj_key]
            prev_position, _ = object_data["positions"][-1]
            
            # Update position history
            object_data["positions"].append((position, timestamp))
            object_data["last_seen"] = timestamp
            
            # Check if object has moved significantly
            distance = ((position[0] - prev_position[0])**2 + (position[1] - prev_position[1])**2)**0.5
            
            # Check if object was on counter and has been moved quickly
            was_on_counter = object_data["in_counter_area"]
            now_on_counter = self._is_in_polygon(position, self.cash_counter_roi)
            
            if was_on_counter and not now_on_counter and not object_data["alert_generated"]:
                # Object was taken from counter - find closest person
                closest_person = self._find_closest_person(position, timestamp)
                if closest_person:
                    object_data["last_associated_person"] = closest_person
                    object_data["alert_generated"] = True
                    alert = self._generate_theft_alert(
                        closest_person, "object_taken", 
                        f"{object_class.capitalize()} {object_id} taken from counter by person {closest_person}",
                        timestamp
                    )
                    if alert:
                        self.suspicious_persons[closest_person] = {
                            "reason": "object_taken",
                            "object": obj_key,
                            "timestamp": timestamp
                        }
                    return alert
            
            # Update state
            object_data["in_counter_area"] = now_on_counter
        
        return None  # No suspicious activity detected
    
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
    
    def _find_closest_person(self, position: Tuple[float, float], timestamp: float, max_distance: float = 0.2) -> Optional[int]:
        """Find the closest person to a given position within a threshold distance"""
        min_distance = float('inf')
        closest_person = None
        
        for obj_id, obj_data in self.tracked_objects.items():
            # Skip non-person objects and only consider recent positions
            if not isinstance(obj_id, int) or obj_data["last_seen"] < timestamp - 1.0:
                continue
            
            # Get person's position closest to the timestamp
            closest_pos = None
            closest_time_diff = float('inf')
            
            for pos, ts in obj_data["positions"]:
                time_diff = abs(ts - timestamp)
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_pos = pos
            
            if closest_pos:
                # Calculate distance
                distance = ((position[0] - closest_pos[0])**2 + (position[1] - closest_pos[1])**2)**0.5
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    closest_person = obj_id
        
        return closest_person
    
    def _analyze_movement_pattern(self, person_id: int) -> float:
        """
        Analyze movement patterns for suspicious behavior.
        Returns a suspicion score between 0 and 1.
        """
        history = self.position_history[person_id]
        
        # Not enough data for analysis
        if len(history) < 10:
            return 0.0
        
        # Calculate metrics that might indicate suspicious behavior
        
        # 1. Excessive back-and-forth movement (pacing)
        position_changes = []
        for i in range(1, len(history)):
            prev_pos, _ = history[i-1]
            curr_pos, _ = history[i]
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            position_changes.append((dx, dy))
        
        # Check for direction reversals
        reversals = 0
        for i in range(1, len(position_changes)):
            prev_dx, prev_dy = position_changes[i-1]
            curr_dx, curr_dy = position_changes[i]
            
            # Check for reversal in x or y direction
            if (prev_dx * curr_dx < 0) or (prev_dy * curr_dy < 0):
                reversals += 1
        
        # 2. Unusually slow or hesitant movement
        speeds = []
        for i in range(1, len(history)):
            (prev_x, prev_y), prev_t = history[i-1]
            (curr_x, curr_y), curr_t = history[i]
            
            dist = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
            time_diff = curr_t - prev_t
            
            if time_diff > 0:
                speed = dist / time_diff
                speeds.append(speed)
        
        avg_speed = sum(speeds) / max(len(speeds), 1)
        speed_variance = sum((s - avg_speed)**2 for s in speeds) / max(len(speeds), 1)
        
        # 3. Observing from a distance
        time_observing = 0
        for (pos, ts) in history:
            # Check if position is near but not in counter area
            in_counter = self._is_in_polygon(pos, self.cash_counter_roi)
            if not in_counter:
                # Calculate distance to counter
                min_dist = float('inf')
                for i in range(len(self.cash_counter_roi)):
                    p1 = self.cash_counter_roi[i]
                    p2 = self.cash_counter_roi[(i+1) % len(self.cash_counter_roi)]
                    dist = self._distance_to_line_segment(pos, p1, p2)
                    min_dist = min(min_dist, dist)
                
                # If within observing distance (but not in counter)
                if 0.1 <= min_dist <= 0.3:
                    time_observing += 1
        
        observing_ratio = time_observing / len(history)
        
        # Combine factors into suspicion score
        reversal_score = min(reversals / 10, 1.0)  # Normalize to 0-1
        speed_score = min(speed_variance * 100, 1.0)  # Unusual speed variations
        observing_score = min(observing_ratio * 2, 1.0)  # Time spent observing
        
        # Weight the factors
        suspicion_score = (0.4 * reversal_score + 0.3 * speed_score + 0.3 * observing_score)
        
        return suspicion_score
    
    def _distance_to_line_segment(self, p: Tuple[float, float], v: Tuple[float, float], w: Tuple[float, float]) -> float:
        """Calculate minimum distance from point p to line segment vw"""
        # Line segment length squared
        l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
        
        # If segment is a point, return distance to point
        if l2 == 0.0:
            return ((p[0] - v[0])**2 + (p[1] - v[1])**2)**0.5
        
        # Project point onto line
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        
        # Calculate projection
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        
        # Return distance to projection
        return ((p[0] - projection[0])**2 + (p[1] - projection[1])**2)**0.5
    
    def _generate_theft_alert(self, person_id: int, alert_type: str, message: str, timestamp: float) -> Dict[str, Any]:
        """Generate theft alert"""
        # Prevent alert flooding
        if timestamp - self.last_alert_time < self.alert_cooldown:
            return None
        
        self.last_alert_time = timestamp
        
        # Create alert
        alert = {
            "alert_id": len(self.theft_incidents) + 1,
            "timestamp": timestamp,
            "person_id": person_id,
            "alert_type": alert_type,
            "message": message,
            "confidence": 0.8,  # Default confidence
            "frame_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        }
        
        # Add alert to history
        self.theft_incidents.append(alert)
        
        # Log alert
        logger.warning(f"THEFT ALERT: {message}")
        
        return alert
    
    def cleanup_stale_objects(self, max_age: float = 30.0):
        """Remove objects that haven't been seen for a while"""
        current_time = time.time()
        to_remove = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data["last_seen"] > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.position_history:
                del self.position_history[obj_id]
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent theft alerts"""
        return sorted(self.theft_incidents, key=lambda x: x["timestamp"], reverse=True)[:count]
    
    def display_alert_overlay(self, frame: np.ndarray, alert: Dict[str, Any]) -> np.ndarray:
        """
        Draw an alert overlay on the frame.
        
        Args:
            frame: OpenCV image in BGR format
            alert: Alert data dictionary
            
        Returns:
            Frame with alert overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for alert
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add alert text
        cv2.putText(frame, f"ALERT: {alert['alert_type'].upper()}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, alert['message'], (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the cash counter and associated zones on the frame.
        
        Args:
            frame: OpenCV image in BGR format
            
        Returns:
            Frame with zones drawn
        """
        h, w = frame.shape[:2]
        
        # Draw cash counter zone
        counter_pts = np.array([
            [int(x * w), int(y * h)] for x, y in self.cash_counter_roi
        ], np.int32)
        cv2.polylines(frame, [counter_pts], True, (0, 255, 255), 2)  # Yellow
        cv2.putText(frame, "Cash Counter", (int(counter_pts[0][0]), int(counter_pts[0][1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw customer zone
        if self.customer_zone is not None:
            customer_pts = np.array([
                [int(x * w), int(y * h)] for x, y in self.customer_zone
            ], np.int32)
            cv2.polylines(frame, [customer_pts], True, (0, 255, 0), 2)  # Green
            cv2.putText(frame, "Customer Zone", (int(customer_pts[0][0]), int(customer_pts[0][1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw cashier zone
        if self.cashier_zone is not None:
            cashier_pts = np.array([
                [int(x * w), int(y * h)] for x, y in self.cashier_zone
            ], np.int32)
            cv2.polylines(frame, [cashier_pts], True, (255, 0, 0), 2)  # Blue
            cv2.putText(frame, "Cashier Zone", (int(cashier_pts[0][0]), int(cashier_pts[0][1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw entry zone
        if self.entry_zone is not None:
            entry_pts = np.array([
                [int(x * w), int(y * h)] for x, y in self.entry_zone
            ], np.int32)
            cv2.polylines(frame, [entry_pts], True, (0, 255, 0), 2)  # Green
            cv2.putText(frame, "Entry Zone", (int(entry_pts[0][0]), int(entry_pts[0][1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw exit zone
        if self.exit_zone is not None:
            exit_pts = np.array([
                [int(x * w), int(y * h)] for x, y in self.exit_zone
            ], np.int32)
            cv2.polylines(frame, [exit_pts], True, (0, 0, 255), 2)  # Red
            cv2.putText(frame, "Exit Zone", (int(exit_pts[0][0]), int(exit_pts[0][1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def handle_person_exit(self, person_id: int, timestamp: float):
        """
        Call this when a person is detected exiting the store. If the person is under suspicion, flag as confirmed theft.
        """
        if person_id in self.suspicious_persons:
            theft_event = {
                "person_id": person_id,
                "reason": self.suspicious_persons[person_id]["reason"],
                "object": self.suspicious_persons[person_id]["object"],
                "timestamp": timestamp,
                "frame_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            }
            self.confirmed_thefts.append(theft_event)
            logger.warning(f"CONFIRMED THEFT: Person {person_id} exited with suspicious activity: {theft_event}")
            del self.suspicious_persons[person_id]
            return theft_event
        return None
    
    def get_confirmed_thefts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent confirmed theft events"""
        return sorted(self.confirmed_thefts, key=lambda x: x["timestamp"], reverse=True)[:count] 