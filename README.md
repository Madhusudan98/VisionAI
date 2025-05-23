# VisionAI - Retail Theft Detection System

A computer vision system designed to detect potential theft activities in retail environments, particularly around cash counter areas. The system integrates person tracking, footfall analysis, and suspicious behavior detection.

## Project Structure

- **hackathon/backend**: Backend server implementation
  - **app/utils/theft_detector.py**: Core implementation of theft detection algorithms
  
- **hackathon/deep_sort_temp**: Deep SORT implementation for object tracking
  - **deepsort.py**: Main deepsort implementation file
  - **deep_sort/**: Deep SORT algorithm components
  - **detector/**: Object detection models (YOLOv3, YOLOv5, Mask R-CNN)
  - **configs/**: Configuration files for detectors and tracking
  - **webserver/**: Web server components for visualization
  
- **hackathon/tools**: Testing and utility scripts
  - **test_theft_detector.py**: Script to test the theft detection system

- **yolov8n.pt**: YOLOv8 nano model weights (already included in the repository)

## Features

1. **Person Tracking**: Tracks individuals throughout the store using Deep SORT algorithm
2. **Footfall Analysis**: Counts entries and exits to monitor store traffic
3. **Suspicious Behavior Detection**:
   - Unusually long dwell times at the cash counter
   - Quick grabs (unusually short interactions at counter)
   - Suspicious movement patterns
   - Unauthorized zone access (customer entering cashier zone)
   - Object theft detection (items taken from counter)

## Setup

The easiest way to set up the project is to run the setup script:

```bash
python setup.py
```

This will:
1. Install required dependencies
2. Check for model weights
3. Create necessary data directories
4. Apply fixes for known issues
5. Check for optional dependencies

## Requirements

Python 3.6+ with the libraries listed in requirements.txt. Install dependencies with:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

Some features require additional packages:

1. **MMDetection**: For using MMDet models
   - Uncomment the mmdet and mmcv-full lines in requirements.txt
   - Run `pip install -r requirements.txt` again

2. **FastReID**: For advanced ReID features
   - Follow the instructions in deep_sort_temp/README.md

## Model Weights

For full functionality, you'll need to download:

1. YOLOv3/YOLOv5 weights for object detection (if using those models)
   - YOLOv8 nano model (yolov8n.pt) is already included in the repository
2. Deep SORT feature extraction model weights
3. (Optional) Mask R-CNN weights for instance segmentation

See the README in the deep_sort_temp directory for specific download links.

## Usage

1. **Test the theft detection system with YOLOv8**:

```bash
python hackathon/tools/test_theft_detector.py --source [VIDEO_PATH] --model yolov8n.pt --show --save
```

2. **Run the Deep SORT tracker with YOLOv3**:

```bash
python hackathon/deep_sort_temp/deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3.yaml --display
```

## Troubleshooting

### Missing 'imghdr' Module

If you encounter a `ModuleNotFoundError: No module named 'imghdr'` error, run:

```bash
python fix_imghdr_issue.py
```

This will modify the code to use PIL instead of imghdr.

### MMDet Import Error

If you encounter a `ModuleNotFoundError: No module named 'mmdet'` error, run:

```bash
python disable_mmdet.py
```

This will make the MMDet import optional. If you need MMDet functionality, install it by uncommenting the relevant lines in requirements.txt.

## How It Works

1. **Object Detection**: Identifies people and objects using YOLO/Mask R-CNN
2. **Object Tracking**: Assigns IDs and tracks objects across frames using Deep SORT
3. **Zone Analysis**: Monitors predefined zones (cash counter, customer area, cashier area)
4. **Behavior Analysis**: Analyzes trajectories, dwell times, and interactions
5. **Alert Generation**: Raises alerts when suspicious patterns are detected

## Alert Types

- **Unusual Dwell Time**: Person stays at counter longer than threshold
- **Quick Grab**: Person interacts with counter unusually quickly
- **Unauthorized Zone**: Customer enters cashier zone
- **Suspicious Movement**: Erratic movement patterns around valuable areas
- **Object Taken**: Item taken from counter by unauthorized person

## Customization

Adjust thresholds and parameters in the TheftDetector class:
- `dwell_time_threshold`: Time considered unusually long at counter (seconds)
- `quick_grab_threshold`: Time considered unusually short for interaction (seconds)
- `suspicious_movement_threshold`: Threshold for suspicious movement detection

## License

See LICENSE file for details. 