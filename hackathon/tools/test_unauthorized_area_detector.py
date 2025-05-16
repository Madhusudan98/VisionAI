#!/usr/bin/env python3
"""
Test Unauthorized Area Detector

A script to demonstrate how to use the unauthorized area detector with label data.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import specialized components
from backend.app.utils.unauthorized_area_detector import UnauthorizedAreaDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test unauthorized area detector")
    parser.add_argument("--source", type=str, default="data/sample_video.mp4", 
                        help="Video source path (default: data/sample_video.mp4)")
    parser.add_argument("--labels", type=str, default="data/labels.csv",
                        help="Path to labels CSV file (default: data/labels.csv)")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show video")
    
    return parser.parse_args()

def setup_video_source(source):
    """Setup video source."""
    # Check if the video file exists
    if not os.path.exists(source):
        print(f"Warning: Video file '{source}' not found. Please provide a valid video file path.")
        # Try to find any video file in the data directory
        data_dir = Path("data")
        if data_dir.exists():
            video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
            if video_files:
                source = str(video_files[0])
                print(f"Using alternative video file: {source}")
            else:
                print("No video files found in data directory. Using webcam as fallback.")
                return cv2.VideoCapture(0)
        else:
            print("Data directory not found. Using webcam as fallback.")
            return cv2.VideoCapture(0)
    
    return cv2.VideoCapture(source)

def setup_video_writer(cap, output_path):
    """Setup video writer if output path is specified."""
    if not output_path:
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def create_sample_labels_file():
    """Create a sample labels file if none exists."""
    labels_dir = Path("data")
    labels_file = labels_dir / "labels.csv"
    
    if not labels_dir.exists():
        os.makedirs(labels_dir)
    
    if not labels_file.exists():
        print("Creating sample labels file...")
        with open(labels_file, 'w') as f:
            f.write("label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height\n")
            f.write("UnAuthorized_Area,248,424,468,499,Screenshot 2025-05-16 at 7.09.25 PM.png,1841,969\n")
        print(f"Sample labels file created at {labels_file}")
    
    return str(labels_file)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Create sample labels file if needed
    if not os.path.exists(args.labels):
        args.labels = create_sample_labels_file()
    
    # Setup video source
    cap = setup_video_source(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Setup video writer
    writer = setup_video_writer(cap, args.output)
    
    # Create unauthorized area detector
    detector = UnauthorizedAreaDetector(labels_file=args.labels)
    
    # Create a simple object tracker (just for demo)
    object_positions = {}
    next_id = 1
    
    print("Testing Unauthorized Area Detector")
    print(f"Loaded {len(detector.areas)} unauthorized areas from {args.labels}")
    print(f"Using video source: {args.source}")
    print("Click on the video to add/move objects")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw unauthorized areas
        frame = detector.draw_areas(frame, color=(0, 0, 255), alpha=0.3)
        
        # Update objects (in a real app, this would come from an object tracker)
        for obj_id, pos in object_positions.items():
            # Draw object
            cv2.circle(frame, pos, 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Person {obj_id}", (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if in unauthorized area
            normalized_pos = (pos[0] / width, pos[1] / height)
            event = detector.update(obj_id, normalized_pos)
            
            # Display alert if unauthorized access detected
            if event:
                print(f"ALERT: {event['message']}")
                cv2.putText(frame, "UNAUTHORIZED ACCESS!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mark object as in unauthorized area if applicable
            if detector.is_object_in_unauthorized_area(obj_id):
                cv2.circle(frame, pos, 15, (0, 0, 255), 2)
                cv2.putText(frame, "UNAUTHORIZED!", (pos[0] - 40, pos[1] + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Click to add/move people", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display recent events
        events = detector.get_recent_events(5)
        y_pos = 90
        for event in events:
            cv2.putText(frame, f"{event['message']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += 20
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if args.show:
            cv2.imshow("Unauthorized Area Detector", frame)
            
            # Handle mouse events for object simulation
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Add new object or move existing one
                    if len(object_positions) < 5:  # Limit to 5 objects for demo
                        nonlocal next_id
                        object_positions[next_id] = (x, y)
                        next_id += 1
                    else:
                        # Move an existing object
                        for obj_id in object_positions:
                            object_positions[obj_id] = (x, y)
                            break
            
            cv2.setMouseCallback("Unauthorized Area Detector", mouse_callback)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 