import cv2
import numpy as np
import time
import os
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ObjectDetector")

class ObjectDetector:
    """
    Object detector class using YOLOv8 for detection and a simple tracker for object tracking.
    """
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, iou_threshold=0.45, classes=None):
        """
        Initialize the object detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            classes (list): List of classes to detect, or None for all classes
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.footfall_tracker = None
        self.next_track_id = 1
        self.tracked_objects = {}  # Dictionary to track objects {id: {bbox, class_id, timestamps, etc.}}
        
        # Load YOLOv8 model
        self._load_model(model_path)
        
        # Get class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else None
        
        logger.info(f"ObjectDetector initialized with model: {model_path}")
    
    def _load_model(self, model_path):
        """
        Load the YOLOv8 model.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            # Attempt to import ultralytics and load YOLOv8 model
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path}")
            self.model_type = "yolov8"
        except ImportError as e:
            logger.error(f"Could not import ultralytics: {e}")
            logger.warning("Falling back to OpenCV DNN for YOLO model")
            
            # Fallback to OpenCV DNN
            self.model = cv2.dnn.readNet(model_path)
            self.model_type = "opencv"
            
            # Try to load class names from coco.names if available
            coco_names_path = os.path.join(os.path.dirname(__file__), "coco.names")
            if os.path.exists(coco_names_path):
                with open(coco_names_path, "r") as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.class_names = [f"class_{i}" for i in range(80)]  # Default COCO classes
    
    def set_footfall_tracker(self, tracker):
        """
        Set a footfall tracker to receive detection updates.
        
        Args:
            tracker: FootfallTracker instance
        """
        self.footfall_tracker = tracker
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: OpenCV image in BGR format
            
        Returns:
            dict: Dictionary with detection results
        """
        if self.model_type == "yolov8":
            return self._detect_yolov8(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_yolov8(self, frame):
        """
        Detect objects using YOLOv8 model.
        
        Args:
            frame: OpenCV image
            
        Returns:
            dict: Detection results
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, classes=self.classes)
        
        # Process results
        detections = {"objects": [], "timestamp": time.time()}
        
        # Process results (assuming first image only)
        if results and len(results) > 0:
            result = results[0]
            
            # Get the detection boxes, confidence scores, and class IDs
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Update object tracking
                    tracking_id = self._update_tracking(class_id, (x1, y1, x2, y2), conf)
                    
                    # Add to detections
                    detections["objects"].append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self.class_names[class_id] if self.class_names else f"class_{class_id}",
                        "tracking_id": tracking_id
                    })
        
        return detections
    
    def _detect_opencv(self, frame):
        """
        Fallback detection method using OpenCV DNN.
        
        Args:
            frame: OpenCV image
            
        Returns:
            dict: Detection results
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Get output layer names
        output_layers = self.model.getUnconnectedOutLayersNames()
        
        # Run forward pass
        outputs = self.model.forward(output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    # Scale bounding box coordinates to image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x1 = int(center_x - w/2)
                    y1 = int(center_y - h/2)
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    # Apply class filter if specified
                    if self.classes is not None and class_id not in self.classes:
                        continue
                    
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS to suppress overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        
        # Prepare detections dict
        detections = {"objects": [], "timestamp": time.time()}
        
        # Process valid detections
        for i in indices:
            # OpenCV 4.5+ returns flat array
            i = i if isinstance(i, int) else i[0]
            
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            class_id = class_ids[i]
            
            # Update object tracking
            tracking_id = self._update_tracking(class_id, (x1, y1, x2, y2), conf)
            
            # Add to detections
            detections["objects"].append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": class_id,
                "class_name": self.class_names[class_id] if self.class_names else f"class_{class_id}",
                "tracking_id": tracking_id
            })
        
        return detections
    
    def _update_tracking(self, class_id, bbox, confidence, max_age=30, iou_threshold=0.3):
        """
        Update object tracking using IOU matching.
        
        Args:
            class_id (int): Class ID of the detected object
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            confidence (float): Detection confidence
            max_age (int): Maximum frames to keep a track without matching
            iou_threshold (float): IOU threshold for track matching
            
        Returns:
            int: Tracking ID
        """
        # Current frame timestamp
        current_time = time.time()
        
        # Calculate box center
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Find the best match among existing tracks
        best_track_id = None
        best_iou = 0
        
        for track_id, track_info in self.tracked_objects.items():
            # Skip tracks with different class_id (except for person class which is more permissive)
            if class_id != track_info["class_id"] and class_id != 0:
                continue
            
            # Calculate IOU with last known bbox
            tx1, ty1, tx2, ty2 = track_info["bbox"]
            
            # Calculate intersection area
            inter_x1 = max(x1, tx1)
            inter_y1 = max(y1, ty1)
            inter_x2 = min(x2, tx2)
            inter_y2 = min(y2, ty2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (tx2 - tx1) * (ty2 - ty1)
                iou = inter_area / float(box1_area + box2_area - inter_area)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
        
        # If a match was found, update the track
        if best_track_id is not None:
            track = self.tracked_objects[best_track_id]
            track["bbox"] = bbox
            track["confidence"] = confidence
            track["last_seen"] = current_time
            track["age"] += 1
            track["positions"].append((center_x, center_y, current_time))
            
            return best_track_id
        
        # No match found, create a new track
        new_track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracked_objects[new_track_id] = {
            "bbox": bbox, 
            "class_id": class_id,
            "confidence": confidence,
            "first_seen": current_time,
            "last_seen": current_time,
            "age": 1,
            "positions": [(center_x, center_y, current_time)]
        }
        
        return new_track_id
    
    def cleanup_stale_tracks(self, max_age_frames=30, max_time_seconds=5.0):
        """
        Remove tracking entries for objects that haven't been seen recently.
        
        Args:
            max_age_frames (int): Maximum frames to keep a track without matching
            max_time_seconds (float): Maximum time in seconds to keep a track without matching
        """
        current_time = time.time()
        stale_ids = []
        
        for track_id, track_info in self.tracked_objects.items():
            time_since_last_seen = current_time - track_info["last_seen"]
            
            if time_since_last_seen > max_time_seconds or track_info["age"] > max_age_frames:
                stale_ids.append(track_id)
        
        # Remove stale tracks
        for track_id in stale_ids:
            del self.tracked_objects[track_id]
    
    def get_tracked_objects(self):
        """
        Get all currently tracked objects.
        
        Returns:
            dict: Dictionary of tracked objects
        """
        return self.tracked_objects

    def reset_tracking(self):
        """Reset all object tracking data"""
        self.tracked_objects = {}
        self.next_track_id = 1 