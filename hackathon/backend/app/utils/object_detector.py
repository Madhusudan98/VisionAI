import cv2
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class for detecting and tracking objects in video frames using YOLOv8.
    Supports integration with footfall tracking.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        """
        Initialize the object detector with YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect, None for all classes
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        self.footfall_tracker = None
        self.next_tracking_id = 1
        self.tracked_objects = {}  # {tracking_id: {history, class_id, etc.}}
        
        # Load YOLOv8 model
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path}")
        except ImportError:
            logger.error("Failed to import ultralytics. Make sure YOLOv8 is installed.")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def set_footfall_tracker(self, tracker):
        """Set the footfall tracker for zone-based analysis"""
        self.footfall_tracker = tracker
        logger.info("Footfall tracker registered with object detector")
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in a frame and track them across frames.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Dictionary with detection results:
            {
                "objects": [
                    {
                        "bbox": [x1, y1, x2, y2],  # Bounding box coordinates
                        "class_id": int,            # Class ID
                        "class_name": str,          # Class name
                        "confidence": float,        # Detection confidence
                        "tracking_id": int          # Unique tracking ID
                    },
                    ...
                ],
                "frame_time": float  # Timestamp of the frame
            }
        """
        start_time = time.time()
        height, width = frame.shape[:2]
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, classes=self.classes, verbose=False)
        
        # Process detections
        objects = []
        current_time = time.time()
        
        # Extract detections from results
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get box coordinates, class, and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Get class name if available
                try:
                    class_name = result.names[class_id]
                except:
                    class_name = f"Class {class_id}"
                
                # Simple tracking based on IOU with previous detections
                tracking_id = self._assign_tracking_id(
                    [x1, y1, x2, y2], class_id, current_time
                )
                
                # Add to objects list
                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "tracking_id": tracking_id
                })
                
                # Update footfall tracker if available
                if self.footfall_tracker:
                    if class_id == 0:  # Assuming class 0 is person
                        # Convert to normalized coordinates for the tracker
                        center_x = (x1 + x2) / (2 * width)
                        center_y = (y1 + y2) / (2 * height)
                        self.footfall_tracker.update_position(tracking_id, (center_x, center_y), current_time)
        
        # Clean up stale tracked objects
        self._cleanup_stale_objects()
        
        return {
            "objects": objects,
            "frame_time": current_time,
            "processing_time": time.time() - start_time
        }
    
    def _assign_tracking_id(self, bbox: List[int], class_id: int, timestamp: float) -> int:
        """
        Assign a tracking ID to a detection based on IOU with previous detections.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID of the detection
            timestamp: Current timestamp
            
        Returns:
            Tracking ID
        """
        # Calculate IoU with all tracked objects of the same class
        max_iou = 0
        best_id = None
        
        for tracking_id, obj_data in self.tracked_objects.items():
            # Only consider objects of the same class and recently seen
            if obj_data["class_id"] != class_id or timestamp - obj_data["last_seen"] > 1.0:
                continue
            
            # Calculate IoU
            prev_bbox = obj_data["bbox"]
            iou = self._calculate_iou(bbox, prev_bbox)
            
            if iou > max_iou and iou > 0.3:  # Minimum IoU threshold for identity
                max_iou = iou
                best_id = tracking_id
        
        if best_id is not None:
            # Update existing tracked object
            self.tracked_objects[best_id]["bbox"] = bbox
            self.tracked_objects[best_id]["last_seen"] = timestamp
            self.tracked_objects[best_id]["history"].append((bbox, timestamp))
            return best_id
        else:
            # Create new tracked object
            new_id = self.next_tracking_id
            self.next_tracking_id += 1
            self.tracked_objects[new_id] = {
                "bbox": bbox,
                "class_id": class_id,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "history": [(bbox, timestamp)]
            }
            return new_id
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _cleanup_stale_objects(self, max_age: float = 5.0):
        """Remove objects that haven't been seen for a while"""
        current_time = time.time()
        to_remove = []
        
        for tracking_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data["last_seen"] > max_age:
                to_remove.append(tracking_id)
        
        for tracking_id in to_remove:
            del self.tracked_objects[tracking_id] 
