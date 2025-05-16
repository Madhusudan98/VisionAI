#!/usr/bin/env python3
"""
Setup script for VisionAI - Retail Theft Detection System
This script helps set up the project by:
1. Installing required dependencies
2. Checking for model weights
3. Applying fixes for known issues
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print_header("Installing dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    return True

def check_model_weights():
    """Check for required model weights"""
    print_header("Checking model weights")
    
    # Check for YOLOv8 weights
    yolov8_path = Path("yolov8n.pt")
    if yolov8_path.exists():
        print(f"✅ Found YOLOv8 weights at {yolov8_path}")
    else:
        print(f"❌ YOLOv8 weights not found at {yolov8_path}")
        print("   Please download YOLOv8 nano weights from:")
        print("   https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    
    # Check for YOLOv3 weights
    yolov3_dir = Path("hackathon/deep_sort_temp/detector/YOLOv3/weight")
    yolov3_path = yolov3_dir / "yolov3.weights"
    
    if not yolov3_dir.exists():
        yolov3_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {yolov3_dir}")
    
    if yolov3_path.exists():
        print(f"✅ Found YOLOv3 weights at {yolov3_path}")
    else:
        print(f"❌ YOLOv3 weights not found at {yolov3_path}")
        print("   To use YOLOv3 with DeepSORT, download weights from:")
        print("   https://pjreddie.com/media/files/yolov3.weights")
    
    # Check for DeepSORT feature extraction model
    deepsort_dir = Path("hackathon/deep_sort_temp/deep_sort/deep/checkpoint")
    deepsort_path = deepsort_dir / "ckpt.t7"
    
    if not deepsort_dir.exists():
        deepsort_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {deepsort_dir}")
    
    if deepsort_path.exists():
        print(f"✅ Found DeepSORT model at {deepsort_path}")
    else:
        print(f"❌ DeepSORT model not found at {deepsort_path}")
        print("   To use DeepSORT, download the model from:")
        print("   https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6")

def fix_known_issues():
    """Apply fixes for known issues"""
    print_header("Applying fixes for known issues")
    
    # Fix for imghdr module issue
    if os.path.exists("fix_imghdr_issue.py"):
        try:
            subprocess.check_call([sys.executable, "fix_imghdr_issue.py"])
            print("✅ Applied fix for imghdr module issue")
        except subprocess.CalledProcessError:
            print("❌ Failed to apply fix for imghdr module issue")
    else:
        print("❓ fix_imghdr_issue.py not found, skipping this fix")
    
    # Fix for MMDet import issue
    if os.path.exists("disable_mmdet.py"):
        try:
            subprocess.check_call([sys.executable, "disable_mmdet.py"])
            print("✅ Applied fix for MMDet import issue")
        except subprocess.CalledProcessError:
            print("❌ Failed to apply fix for MMDet import issue")
    else:
        print("❓ disable_mmdet.py not found, skipping this fix")

def create_data_directories():
    """Create necessary data directories"""
    print_header("Creating data directories")
    
    dirs = [
        "data/theft",
        "data/footfall",
        "outputs"
    ]
    
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {path}")
        else:
            print(f"✓ Directory already exists: {path}")

def check_optional_dependencies():
    """Check for optional dependencies"""
    print_header("Checking optional dependencies")
    
    # Check for MMDet
    try:
        import mmdet
        print("✅ MMDet is installed")
    except ImportError:
        print("❓ MMDet is not installed. This is optional and only needed if you want to use MMDet models.")
        print("   To install MMDet, uncomment the mmdet and mmcv-full lines in requirements.txt and run:")
        print("   pip install -r requirements.txt")
    
    # Check for FastReID (another optional dependency)
    try:
        import fastreid
        print("✅ FastReID is installed")
    except ImportError:
        print("❓ FastReID is not installed. This is optional and only needed for advanced ReID features.")
        print("   To install FastReID, follow the instructions in the deep_sort_temp/README.md file.")

def main():
    """Main setup function"""
    print_header("VisionAI - Retail Theft Detection System Setup")
    
    install_dependencies()
    check_model_weights()
    create_data_directories()
    fix_known_issues()
    check_optional_dependencies()
    
    print_header("Setup Complete")
    print("""
You can now run the system using:

1. Test the theft detection system with YOLOv8:
   python hackathon/tools/test_theft_detector.py --source [VIDEO_PATH] --model yolov8n.pt --show --save

2. Run the Deep SORT tracker with YOLOv3 (after downloading weights):
   python hackathon/deep_sort_temp/deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3.yaml --display

For more information, see the README.md file.
""")

if __name__ == "__main__":
    main() 