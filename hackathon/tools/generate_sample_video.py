#!/usr/bin/env python3
"""
Generate Sample Video

A script to generate a sample video file with moving objects for testing the detector modules.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def generate_sample_video(output_path, width=640, height=480, duration=10, fps=30):
    """
    Generate a sample video with moving objects.
    
    Args:
        output_path: Path to save the video
        width: Video width
        height: Video height
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create moving objects
    objects = [
        {
            'id': 1,
            'position': (50, height // 2),
            'velocity': (5, 0),
            'color': (0, 0, 255),  # Red
            'radius': 15
        },
        {
            'id': 2,
            'position': (width - 50, height // 2 - 50),
            'velocity': (-3, 1),
            'color': (0, 255, 0),  # Green
            'radius': 12
        },
        {
            'id': 3,
            'position': (width // 2, height - 50),
            'velocity': (0, -4),
            'color': (255, 0, 0),  # Blue
            'radius': 18
        }
    ]
    
    # Define zones
    zones = {
        "cash_counter": np.array([
            [0.4, 0.3],  # top-left
            [0.6, 0.3],  # top-right
            [0.6, 0.4],  # bottom-right
            [0.4, 0.4]   # bottom-left
        ]),
        "entrance": np.array([
            [0.0, 0.4],
            [0.2, 0.4],
            [0.2, 0.6],
            [0.0, 0.6]
        ]),
        "exit": np.array([
            [0.8, 0.4],
            [1.0, 0.4],
            [1.0, 0.6],
            [0.8, 0.6]
        ])
    }
    
    # Generate frames
    total_frames = duration * fps
    for i in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw zones
        for zone_id, polygon in zones.items():
            points = []
            for x, y in polygon:
                points.append((int(x * width), int(y * height)))
            
            if zone_id == "cash_counter":
                color = (0, 0, 255)  # Red
            elif zone_id == "entrance":
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue
            
            cv2.polylines(frame, [np.array(points)], True, color, 2)
            
            # Calculate centroid for label
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Draw zone ID
            cv2.putText(frame, zone_id, (int(centroid_x), int(centroid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw entry/exit lines
        cv2.line(frame, (0, height // 2), (int(width * 0.2), height // 2), (0, 255, 0), 2)
        cv2.putText(frame, "entry_line", (int(width * 0.1), height // 2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.line(frame, (int(width * 0.8), height // 2), (width, height // 2), (0, 0, 255), 2)
        cv2.putText(frame, "exit_line", (int(width * 0.9), height // 2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Update and draw objects
        for obj in objects:
            # Update position
            obj['position'] = (
                obj['position'][0] + obj['velocity'][0],
                obj['position'][1] + obj['velocity'][1]
            )
            
            # Bounce off walls
            if obj['position'][0] <= 0 or obj['position'][0] >= width:
                obj['velocity'] = (-obj['velocity'][0], obj['velocity'][1])
            if obj['position'][1] <= 0 or obj['position'][1] >= height:
                obj['velocity'] = (obj['velocity'][0], -obj['velocity'][1])
            
            # Draw object
            cv2.circle(frame, obj['position'], obj['radius'], obj['color'], -1)
            cv2.putText(frame, f"ID: {obj['id']}", 
                       (obj['position'][0] + 10, obj['position'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj['color'], 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Sample video generated at {output_path}")

if __name__ == "__main__":
    # Set output path
    output_path = Path(__file__).resolve().parent.parent / "data" / "sample_video.mp4"
    
    # Generate sample video
    generate_sample_video(str(output_path), duration=30) 