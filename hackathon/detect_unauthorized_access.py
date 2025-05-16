#!/usr/bin/env python3
"""
Detect Unauthorized Access

A convenient script to run the unauthorized area detector on any video file.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect unauthorized access in a video")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--labels", type=str, default="data/labels.csv", 
                        help="Path to labels CSV file (default: data/labels.csv)")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Don't display video")
    
    args = parser.parse_args()
    
    # Convert to absolute paths if needed
    video_path = os.path.abspath(args.video_path)
    labels_path = os.path.abspath(args.labels)
    output_path = os.path.abspath(args.output) if args.output else ""
    
    # Build the command to run the detector
    cmd = [
        "python", 
        "tools/test_unauthorized_area_detector.py",
        f"--source={video_path}",
        f"--labels={labels_path}"
    ]
    
    if output_path:
        cmd.append(f"--output={output_path}")
    
    if not args.no_display:
        cmd.append("--show")
    
    # Print the command
    print("Running command:", " ".join(cmd))
    
    # Run the command
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main() 