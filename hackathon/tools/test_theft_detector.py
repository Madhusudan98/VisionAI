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
from backend.app.utils.roi_selector import ROISelector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source (file path, URL, or webcam index)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--use-default-roi', action='store_true', help='Use default ROI instead of interactive selection')
    return parser.parse_args()

def get_default_roi():
    """Return default ROI coordinates if user opts to skip selection"""
    # Define zones for footfall tracking (normalized coordinates 0-1)
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
    
    # Define cash counter area (in middle of frame)
    cash_counter_roi = np.array([
        [0.4, 0.45],  # top-left
        [0.6, 0.45],  # top-right
        [0.6, 0.55],  # bottom-right
        [0.4, 0.55]   # bottom-left
    ])
    
    # Define customer zone (in front of counter)
    customer_zone = np.array([
        [0.35, 0.55],  # top-left
        [0.65, 0.55],  # top-right
        [0.65, 0.75],  # bottom-right
        [0.35, 0.75]   # bottom-left
    ])
    
    # Define cashier zone (behind counter)
    cashier_zone = np.array([
        [0.35, 0.25],  # top-left
        [0.65, 0.25],  # top-right
        [0.65, 0.45],  # bottom-right
        [0.35, 0.45]   # bottom-left
    ])
    
    return {
        "entry_zone": entry_zone,
        "exit_zone": exit_zone,
        "cash_counter": cash_counter_roi,
        "cashier_zone": cashier_zone,
        "customer_zone": customer_zone
    }

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
    
    # Open the video source to get the first frame for ROI selection
    try:
        source = args.source
        if source.isnumeric():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
            
        # Read the first frame for ROI selection
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to grab the first frame from the video source.")
            return
            
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Interactive ROI selection or use defaults
    if args.use_default_roi:
        print("Using default regions of interest")
        regions = get_default_roi()
    else:
        print("Please select regions of interest on the first frame:")
        selector = ROISelector(window_name="Select Regions of Interest")
        regions = selector.select_roi(
            first_frame, 
            ["cash_counter", "entry_zone", "exit_zone", "cashier_zone", "customer_zone"]
        )
    
    # Create ROI zones dictionary with mapping to IDs
    roi_zones = {
        1: regions["entry_zone"],  # Entry zone ID = 1
        2: regions["exit_zone"],   # Exit zone ID = 2
        3: regions["cash_counter"]  # Cash counter zone ID = 3
    }
    
    # Initialize footfall tracker
    footfall_dir = os.path.join("data", "footfall")
    os.makedirs(footfall_dir, exist_ok=True)
    
    footfall_tracker = FootfallTracker(
        roi_zones=roi_zones,
        entry_roi_id=1,
        exit_roi_id=2,
        data_dir=footfall_dir,
    )
    
    # Initialize theft detector
    theft_dir = os.path.join("data", "theft")
    os.makedirs(theft_dir, exist_ok=True)
    
    theft_detector = TheftDetector(
        cash_counter_roi=regions["cash_counter"],
        dwell_time_threshold=20.0,  # Lower for demo purposes
        quick_grab_threshold=2.0,   # Lower for demo purposes
        data_dir=theft_dir,
        entry_zone=regions["entry_zone"],
        exit_zone=regions["exit_zone"],
        customer_zone=regions["customer_zone"],
        cashier_zone=regions["cashier_zone"]
    )
    
    # Register the tracker with the detector
    detector.set_footfall_tracker(footfall_tracker)
    
    # Reset video capture to start 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if saving
    out = None
    if args.save:
        output_path = os.path.join(args.output_dir, f"theft_detection_{int(time.time())}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Process video frames
    frame_count = 0
    start_time = time.time()
    current_alert = None
    alert_start_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw zones
        frame = theft_detector.draw_zones(frame)
        
        # Track objects and check for theft
        for obj in detections["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            class_id = obj["class_id"]
            tracking_id = obj["tracking_id"]
            
            # Convert to normalized coordinates
            center_x = (x1 + x2) / (2 * width)
            center_y = (y1 + y2) / (2 * height)
            
            # Draw bounding box
            if class_id == 0:  # Person
                color = (0, 255, 0)  # Green
                label = f"Person {tracking_id}"
                
                # Update theft detector for person
                alert = theft_detector.update_person_position(tracking_id, (center_x, center_y), current_time)
                if alert and not current_alert:
                    current_alert = alert
                    alert_start_time = current_time
                    
                # Check if the person has exited
                person_data = footfall_tracker.get_person_data(tracking_id)
                if person_data and "simple_transitions" in person_data:
                    # Check for exit zone transitions (using the simple format)
                    for transition_type, timestamp in person_data["simple_transitions"]:
                        if transition_type == "exit":
                            # Person has exited, check for theft
                            theft_event = theft_detector.handle_person_exit(tracking_id, current_time)
                            if theft_event:
                                print(f"CONFIRMED THEFT: Person {tracking_id} exited after suspicious activity!")
            else:
                color = (255, 0, 0)  # Red for other objects
                class_name = obj.get("class_name", f"Class {class_id}")
                label = f"{class_name} {tracking_id}"
                
                # Track objects that might be stolen (phones, bags, laptops, etc.)
                object_classes = ["cell phone", "backpack", "handbag", "suitcase", "laptop", "wallet"]
                if class_name.lower() in object_classes:
                    alert = theft_detector.update_object_position(
                        tracking_id, class_name.lower(), (center_x, center_y), current_time
                    )
                    if alert and not current_alert:
                        current_alert = alert
                        alert_start_time = current_time
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display footfall stats
        footfall_stats = footfall_tracker.get_current_footfall_count()
        
        # Draw transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw stats
        cv2.putText(frame, f"Current Visitors: {footfall_stats['current_visitors']}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Entries: {footfall_stats['total_entries']}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Exits: {footfall_stats['total_exits']}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw alerts if any
        if current_alert:
            frame = theft_detector.display_alert_overlay(frame, current_alert)
            
            # Clear alert after 5 seconds
            if current_time - alert_start_time > 5.0:
                current_alert = None
        
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
        theft_detector.cleanup_stale_objects()
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")
    print(f"Footfall stats: {footfall_tracker.get_current_footfall_count()}")
    
    # Print theft alerts
    print("\nSuspicious Activity Alerts:")
    recent_alerts = theft_detector.get_recent_alerts()
    if recent_alerts:
        for alert in recent_alerts:
            print(f"[{alert['frame_time']}] {alert['alert_type']}: {alert['message']}")
    else:
        print("No suspicious activity detected.")
        
    # Print confirmed thefts
    print("\nConfirmed Theft Events:")
    confirmed_thefts = theft_detector.get_confirmed_thefts()
    if confirmed_thefts:
        for theft in confirmed_thefts:
            print(f"[{theft['frame_time']}] Person {theft['person_id']} - Reason: {theft['reason']}")
    else:
        print("No confirmed thefts detected.")

if __name__ == "__main__":
    main() 