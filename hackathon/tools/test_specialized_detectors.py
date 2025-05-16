#!/usr/bin/env python3
"""
Test Specialized Detectors

A script to demonstrate how to use the specialized detector components.
This shows how to use each component individually or the integration class.
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
from backend.app.utils.line_crossing_detector import LineCrossingDetector
from backend.app.utils.visitor_counter import VisitorCounter
from backend.app.utils.zone_detector import ZoneDetector
from backend.app.utils.dwell_time_analyzer import DwellTimeAnalyzer
from backend.app.utils.movement_anomaly_detector import MovementAnomalyDetector
from backend.app.utils.detector_integration import DetectorIntegration

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test specialized detectors")
    parser.add_argument("--source", type=str, default="data/sample_video.mp4", 
                        help="Video source path (default: data/sample_video.mp4)")
    parser.add_argument("--mode", type=str, default="integration", 
                        choices=["line", "visitor", "zone", "dwell", "anomaly", "integration"],
                        help="Which detector to test")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show video")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
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

def test_line_crossing_detector(args):
    """Test the line crossing detector."""
    print("Testing Line Crossing Detector")
    
    # Setup video source
    cap = setup_video_source(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Setup video writer
    writer = setup_video_writer(cap, args.output)
    
    # Create line crossing detector
    entry_line = LineCrossingDetector(
        line_start=(0.1, 0.5),
        line_end=(0.4, 0.5),
        line_id="entry_line",
        crossing_direction="both"
    )
    
    # Create a simple object tracker (just for demo)
    # In a real application, you would use a proper object tracker
    object_positions = {}
    next_id = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Simulate object tracking with mouse clicks
        cv2.putText(frame, "Click to add/move objects", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw line
        frame = entry_line.draw_line(frame)
        
        # Update objects (in a real app, this would come from an object tracker)
        for obj_id, pos in object_positions.items():
            # Draw object
            cv2.circle(frame, pos, 10, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update line crossing detector
            normalized_pos = (pos[0] / width, pos[1] / height)
            event = entry_line.update(obj_id, normalized_pos)
            
            # Display event if line was crossed
            if event:
                print(f"Line crossed: {event}")
                cv2.putText(frame, f"Line crossed by {obj_id}!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display recent crossings
        crossings = entry_line.get_recent_crossings(5)
        y_pos = 90
        for crossing in crossings:
            cv2.putText(frame, f"ID {crossing['object_id']} crossed at {crossing['timestamp']:.1f}s", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if args.show:
            cv2.imshow("Line Crossing Detector", frame)
            
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
            
            cv2.setMouseCallback("Line Crossing Detector", mouse_callback)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def test_zone_detector(args):
    """Test the zone detector."""
    print("Testing Zone Detector")
    
    # Setup video source
    cap = setup_video_source(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Setup video writer
    writer = setup_video_writer(cap, args.output)
    
    # Define zones
    zones = {
        "cash_counter": np.array([
            [0.4, 0.3],  # top-left
            [0.6, 0.3],  # top-right
            [0.6, 0.4],  # bottom-right
            [0.4, 0.4]   # bottom-left
        ]),
        "entrance": np.array([
            [0.1, 0.4],
            [0.3, 0.4],
            [0.3, 0.6],
            [0.1, 0.6]
        ]),
        "exit": np.array([
            [0.7, 0.4],
            [0.9, 0.4],
            [0.9, 0.6],
            [0.7, 0.6]
        ])
    }
    
    # Create zone detector
    zone_detector = ZoneDetector(zones=zones)
    
    # Create a simple object tracker (just for demo)
    object_positions = {}
    next_id = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Simulate object tracking with mouse clicks
        cv2.putText(frame, "Click to add/move objects", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw zones
        frame = zone_detector.draw_zones(frame)
        
        # Update objects
        for obj_id, pos in object_positions.items():
            # Draw object
            cv2.circle(frame, pos, 10, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update zone detector
            normalized_pos = (pos[0] / width, pos[1] / height)
            event = zone_detector.update(obj_id, normalized_pos)
            
            # Display event if zone was entered/exited
            if event:
                print(f"Zone event: {event}")
                cv2.putText(frame, f"Zone {event['transition_type']} by {obj_id}!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display which zones this object is in
            zones_in = zone_detector.get_zones_for_object(obj_id)
            if zones_in:
                cv2.putText(frame, f"In zones: {', '.join(zones_in)}", 
                           (pos[0] + 10, pos[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display zone occupancy
        y_pos = 90
        for zone_id in zones:
            objects = zone_detector.get_objects_in_zone(zone_id)
            cv2.putText(frame, f"Zone {zone_id}: {len(objects)} objects", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if args.show:
            cv2.imshow("Zone Detector", frame)
            
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
            
            cv2.setMouseCallback("Zone Detector", mouse_callback)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def test_integration(args):
    """Test the detector integration."""
    print("Testing Detector Integration")
    
    # Setup video source
    cap = setup_video_source(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Setup video writer
    writer = setup_video_writer(cap, args.output)
    
    # Create detector integration
    integration = DetectorIntegration(store_id="test_store")
    
    # Create a simple object tracker (just for demo)
    object_positions = {}
    object_trajectories = {}
    next_id = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Simulate object tracking with mouse clicks
        cv2.putText(frame, "Click to add/move objects", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw visualization
        frame = integration.draw_visualization(frame)
        
        # Update objects
        for obj_id, pos in object_positions.items():
            # Draw object
            cv2.circle(frame, pos, 10, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update trajectory
            if obj_id not in object_trajectories:
                object_trajectories[obj_id] = []
            object_trajectories[obj_id].append(pos)
            
            # Limit trajectory length
            if len(object_trajectories[obj_id]) > 50:
                object_trajectories[obj_id] = object_trajectories[obj_id][-50:]
            
            # Draw trajectory
            for i in range(len(object_trajectories[obj_id]) - 1):
                p1 = object_trajectories[obj_id][i]
                p2 = object_trajectories[obj_id][i + 1]
                alpha = (i + 1) / len(object_trajectories[obj_id])
                color = (int(255 * (1 - alpha)), 0, int(255 * alpha))
                cv2.line(frame, p1, p2, color, 2)
            
            # Update detector integration
            normalized_pos = (pos[0] / width, pos[1] / height)
            alerts = integration.update(obj_id, normalized_pos)
            
            # Display alerts
            if alerts:
                for alert in alerts:
                    print(f"Alert: {alert}")
                    alert_type = alert.get("transition_type", alert.get("alert_type", "unknown"))
                    cv2.putText(frame, f"Alert: {alert_type}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if args.show:
            cv2.imshow("Detector Integration", frame)
            
            # Handle mouse events for object simulation
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Add new object
                    if len(object_positions) < 5:  # Limit to 5 objects for demo
                        nonlocal next_id
                        object_positions[next_id] = (x, y)
                        next_id += 1
                elif event == cv2.EVENT_MOUSEMOVE:
                    # Move the last added object if mouse button is held down
                    if flags & cv2.EVENT_FLAG_LBUTTON and object_positions:
                        last_id = max(object_positions.keys())
                        object_positions[last_id] = (x, y)
            
            cv2.setMouseCallback("Detector Integration", mouse_callback)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Run the selected test
    if args.mode == "line":
        test_line_crossing_detector(args)
    elif args.mode == "zone":
        test_zone_detector(args)
    elif args.mode == "integration":
        test_integration(args)
    else:
        print(f"Test mode '{args.mode}' not implemented yet")

if __name__ == "__main__":
    main() 