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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source (file path, URL, or webcam index)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    return parser.parse_args()

def draw_zones(frame, entry_zone, exit_zone, height, width):
    """Draw entry and exit zones on the frame"""
    # Convert normalized coordinates to pixel coordinates
    entry_points = [(int(x * width), int(y * height)) for x, y in entry_zone]
    exit_points = [(int(x * width), int(y * height)) for x, y in exit_zone]
    
    # Draw entry zone (green)
    cv2.polylines(frame, [np.array(entry_points)], True, (0, 255, 0), 2)
    cv2.putText(frame, "Entry Zone", entry_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw exit zone (red)
    cv2.polylines(frame, [np.array(exit_points)], True, (0, 0, 255), 2)
    cv2.putText(frame, "Exit Zone", exit_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def draw_footfall_stats(frame, tracker):
    """Display footfall statistics on the frame"""
    stats = tracker.get_current_footfall_count()
    h, w = frame.shape[:2]
    
    # Draw transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw stats
    cv2.putText(frame, f"Current Visitors: {stats['current_visitors']}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Entries: {stats['total_entries']}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Exits: {stats['total_exits']}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
        classes=None  # Detect all classes, we'll filter for persons in tracking
    )
    
    # Define ROI zones (normalized coordinates 0-1)
    # Left side of the frame is entry, right side is exit
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
    
    # Create ROI zones dictionary with mapping to IDs
    roi_zones = {
        1: entry_zone,  # Entry zone ID = 1
        2: exit_zone    # Exit zone ID = 2
    }
    
    # Initialize footfall tracker
    data_dir = os.path.join("data", "footfall")
    os.makedirs(data_dir, exist_ok=True)
    
    tracker = FootfallTracker(
        roi_zones=roi_zones,
        entry_roi_id=1,
        exit_roi_id=2,
        data_dir=data_dir
    )
    
    # Register the tracker with the detector
    detector.set_footfall_tracker(tracker)
    
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw detections
        for obj in detections["objects"]:
            if obj["class_id"] == 0:  # Person class
                x1, y1, x2, y2 = obj["bbox"]
                label = f"Person {obj['tracking_id']}: {obj['confidence']:.2f}"
                
                # Convert to normalized coordinates for footfall tracking
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                
                # Update footfall tracker
                tracker.update_object_position(obj["tracking_id"], (center_x, center_y))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw zones and stats
        frame = draw_zones(frame, entry_zone, exit_zone, height, width)
        frame = draw_footfall_stats(frame, tracker)
        
        # Add processing information
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show results
        if args.show:
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Save results
        if args.save and out:
            out.write(frame)
        
        # Clean up stale objects
        tracker.cleanup_stale_objects()
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")
    print(f"Footfall stats: {tracker.get_current_footfall_count()}")

if __name__ == "__main__":
    main() 