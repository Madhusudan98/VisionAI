#!/usr/bin/env python
"""
Runner script for Theft Detector application.
Simplifies running the theft detector with various options.
"""

import os
import sys
import subprocess

def print_header():
    print("=" * 60)
    print("          RETAIL THEFT DETECTION SYSTEM          ")
    print("=" * 60)
    print("This application detects suspicious behaviors and potential")
    print("theft incidents in retail environments.")
    print()

def print_usage():
    print("Usage:")
    print("  python run_theft_detector.py [options]")
    print()
    print("Options:")
    print("  --video <path>     : Path to video file (required)")
    print("  --interactive      : Enable interactive ROI selection (default)")
    print("  --default-roi      : Use default ROIs instead of interactive selection")
    print("  --save             : Save the processed video")
    print("  --no-display       : Don't show the video during processing")
    print("  --confidence <val> : Detection confidence threshold (0.0-1.0, default: 0.5)")
    print("  --model <path>     : Path to YOLOv8 model file (default: yolov8n.pt)")
    print("  --help             : Show this help message")
    print()

def main():
    print_header()
    
    # Parse simple arguments
    args = sys.argv[1:]
    if not args or "--help" in args:
        print_usage()
        return
    
    # Check for video source
    video_source = None
    for i, arg in enumerate(args):
        if arg == "--video" and i+1 < len(args):
            video_source = args[i+1]
            break
    
    if not video_source:
        print("Error: No video source specified. Use --video <path>")
        print()
        print_usage()
        return
    
    # Build command
    cmd = [sys.executable, "hackathon/tools/test_theft_detector.py", "--source", video_source]
    
    # Add optional arguments
    if "--default-roi" in args:
        cmd.append("--use-default-roi")
    
    if "--save" in args:
        cmd.append("--save")
    
    if "--no-display" not in args:
        cmd.append("--show")
    
    for i, arg in enumerate(args):
        if arg == "--confidence" and i+1 < len(args):
            cmd.extend(["--conf-thres", args[i+1]])
        elif arg == "--model" and i+1 < len(args):
            cmd.extend(["--model", args[i+1]])
    
    print(f"Running theft detector with video: {video_source}")
    print(f"Full command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTheft detection interrupted by user.")
    except Exception as e:
        print(f"\nError running theft detector: {e}")

if __name__ == "__main__":
    main() 