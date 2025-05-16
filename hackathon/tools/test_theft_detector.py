import cv2
import numpy as np
import sys
import os
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.utils.object_detector import ObjectDetector
from backend.app.utils.footfall_tracker import FootfallTracker
from backend.app.utils.theft_detector import TheftDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source (file path, URL, or webcam index)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    return parser.parse_args()

def draw_zones(frame, entry_zone, exit_zone, cash_counter_zone, height, width):
    """Draw all zones on the frame"""
    # Convert normalized coordinates to pixel coordinates
    entry_points = [(int(x * width), int(y * height)) for x, y in entry_zone]
    exit_points = [(int(x * width), int(y * height)) for x, y in exit_zone]
    cash_counter_points = [(int(x * width), int(y * height)) for x, y in cash_counter_zone]
    
    # Draw entry zone (green)
    cv2.polylines(frame, [np.array(entry_points)], True, (0, 255, 0), 2)
    cv2.putText(frame, "Entry Zone", entry_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw exit zone (red)
    cv2.polylines(frame, [np.array(exit_points)], True, (0, 0, 255), 2)
    cv2.putText(frame, "Exit Zone", exit_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw cash counter zone (yellow)
    cv2.polylines(frame, [np.array(cash_counter_points)], True, (0, 255, 255), 2)
    cv2.putText(frame, "Cash Counter", cash_counter_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def draw_stats(frame, footfall_tracker, theft_detector):
    """Display combined statistics on the frame"""
    footfall_stats = footfall_tracker.get_current_footfall_count()
    theft_alerts = theft_detector.get_alerts()
    
    # Draw transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw footfall stats
    cv2.putText(frame, f"Current Visitors: {footfall_stats['current_visitors']}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Entries: {footfall_stats['total_entries']}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Exits: {footfall_stats['total_exits']}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw theft alerts indicator
    alert_color = (0, 0, 255) if len(theft_alerts) > 0 else (255, 255, 255)
    cv2.putText(frame, f"Theft Alerts: {len(theft_alerts)}", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
    
    # Display latest alert if exists
    if theft_alerts:
        latest_alert = theft_alerts[-1]
        cv2.putText(frame, f"ALERT: {latest_alert['type']}", (width - 350, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    args = parse_args()
    
    # Create output directory if saving results
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.conf_thres,
        classes=None  # Detect all classes
    )
    
    # Define ROI zones (normalized coordinates 0-1)
    entry_zone = np.array([
        [0.0, 0.0],   # top-left
        [0.4, 0.0],   # top-right
        [0.4, 1.0],   # bottom-right
        [0.0, 1.0]    # bottom-left
    ])
    
    exit_zone = np.array([
        [0.6, 0.0],   # top-left
        [1.0, 0.0],   # top-right
        [1.0, 1.0],   # bottom-right
        [0.6, 1.0]    # bottom-left
    ])
    
    # Define cash counter zone (in the middle bottom of the frame)
    cash_counter_zone = np.array([
        [0.4, 0.6],   # top-left
        [0.6, 0.6],   # top-right
        [0.6, 1.0],   # bottom-right
        [0.4, 1.0]    # bottom-left
    ])
    
    # Create ROI zones dictionary with mapping to IDs
    roi_zones = {
        1: entry_zone,         # Entry zone ID = 1
        2: exit_zone,          # Exit zone ID = 2
        3: cash_counter_zone   # Cash counter zone ID = 3
    }
    
    # Initialize footfall tracker
    data_dir = os.path.join("data", "footfall")
    os.makedirs(data_dir, exist_ok=True)
    
    footfall_tracker = FootfallTracker(
        roi_zones=roi_zones,
        entry_roi_id=1,
        exit_roi_id=2,
        data_dir=data_dir
    )
    
    # Initialize theft detector
    theft_detector = TheftDetector(
        cash_counter_roi=cash_counter_zone,
        dwell_time_threshold=5.0,      # 5 seconds dwell time threshold
        quick_grab_threshold=1.0,      # 1 second quick grab threshold
        suspicious_movement_threshold=0.2  # Movement threshold for suspicious activity
    )
    
    # Register the trackers with the detector
    detector.set_footfall_tracker(footfall_tracker)
    
    # Initialize video capture
    try:
        source = args.source
        if source.isnumeric():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if saving
    out = None
    if args.save:
        output_path = os.path.join(args.output_dir, f"output_{int(time.time())}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Process video frames
    frame_count = 0
    start_time = time.time()
    theft_alert_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        detections = detector.detect(frame)
        
        # Process detections and update trackers
        for obj in detections["objects"]:
            if obj["class_id"] == 0:  # Person class
                x1, y1, x2, y2 = obj["bbox"]
                label = f"Person {obj['tracking_id']}: {obj['confidence']:.2f}"
                
                # Convert to normalized coordinates
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                
                # Update footfall tracker
                footfall_tracker.update_object_position(obj['tracking_id'], (center_x, center_y))
                
                # Update theft detector (with additional person data like size)
                person_height = (y2 - y1) / height
                person_width = (x2 - x1) / width
                person_size = person_width * person_height
                
                theft_detector.update_person_position(
                    obj['tracking_id'], 
                    (center_x, center_y), 
                    size=person_size, 
                    timestamp=time.time()
                )
                
                # Draw bounding box
                box_color = (0, 255, 0)  # Default green
                
                # Check if this person is in cash counter zone
                in_cash_counter = False
                for roi_id, roi in roi_zones.items():
                    if roi_id == 3:  # Cash counter zone
                        if cv2.pointPolygonTest(roi * np.array([width, height]), (int(center_x * width), int(center_y * height)), False) >= 0:
                            in_cash_counter = True
                            box_color = (0, 165, 255)  # Orange for cash counter zone
                            break
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            elif obj["class_id"] in [39, 40, 41, 67, 73]:  # Common objects that might be on a counter (bottle, cup, cell phone, laptop, book)
                x1, y1, x2, y2 = obj["bbox"]
                
                # Get class name
                class_names = {
                    39: "Bottle", 
                    40: "Wine Glass", 
                    41: "Cup", 
                    67: "Cell Phone", 
                    73: "Book"
                }
                class_name = class_names.get(obj["class_id"], f"Object {obj['class_id']}")
                
                label = f"{class_name} {obj['tracking_id']}: {obj['confidence']:.2f}"
                
                # Convert to normalized coordinates
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                
                # Update theft detector with object position
                theft_detector.update_object_position(
                    obj['tracking_id'], 
                    class_name,
                    (center_x, center_y), 
                    timestamp=time.time()
                )
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for objects
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Check for theft alerts
        theft_alerts = theft_detector.get_alerts()
        if len(theft_alerts) > theft_alert_count:
            # New alert detected
            alert = theft_alerts[-1]
            print(f"THEFT ALERT: {alert['type']} at {alert['timestamp']}")
            theft_alert_count = len(theft_alerts)
        
        # Draw zones and stats
        frame = draw_zones(frame, entry_zone, exit_zone, cash_counter_zone, height, width)
        frame = draw_stats(frame, footfall_tracker, theft_detector)
        
        # Add processing information
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show results
        if args.show:
            cv2.imshow("Theft Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Save results
        if args.save and out:
            out.write(frame)
        
        # Clean up stale objects
        footfall_tracker.cleanup_stale_objects()
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")
    print(f"Footfall stats: {footfall_tracker.get_current_footfall_count()}")
    print(f"Theft alerts: {theft_alert_count}")

if __name__ == "__main__":
    main() 