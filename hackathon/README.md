# VisionAI - Specialized Detector Modules

This project contains specialized computer vision detector modules for retail theft detection and visitor analysis. Each module is designed with a single responsibility, making them modular, maintainable, and reusable.

## Project Structure

```
hackathon/
├── backend/
│   └── app/
│       └── utils/
│           ├── line_crossing_detector.py   # Detects objects crossing boundaries
│           ├── visitor_counter.py          # Counts and tracks store visitors
│           ├── zone_detector.py            # Monitors objects in defined zones
│           ├── dwell_time_analyzer.py      # Analyzes how long objects stay in areas
│           ├── movement_anomaly_detector.py # Detects unusual movement patterns
│           ├── unauthorized_area_detector.py # Detects access to unauthorized areas
│           ├── detector_integration.py     # Integrates all detector modules
│           └── README.md                   # Detailed documentation for each module
├── data/
│   ├── sample_video.mp4                    # Sample video for testing
│   └── labels.csv                          # Label data for unauthorized areas
└── tools/
    ├── test_specialized_detectors.py       # Script to test the detector modules
    ├── test_unauthorized_area_detector.py  # Script to test unauthorized area detection
    └── generate_sample_video.py            # Script to generate sample test videos
```

## Getting Started

1. **Setup Environment**

   Make sure you have Python 3.8+ and OpenCV installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Video** (if needed)

   ```bash
   python hackathon/tools/generate_sample_video.py
   ```

3. **Test the Detector Modules**

   ```bash
   # Test line crossing detector
   python hackathon/tools/test_specialized_detectors.py --mode line --source hackathon/data/sample_video.mp4 --show

   # Test zone detector
   python hackathon/tools/test_specialized_detectors.py --mode zone --source hackathon/data/sample_video.mp4 --show

   # Test the integrated system
   python hackathon/tools/test_specialized_detectors.py --mode integration --source hackathon/data/sample_video.mp4 --show
   
   # Test unauthorized area detection
   python hackathon/tools/test_unauthorized_area_detector.py --source hackathon/data/sample_video.mp4 --show
   ```

## Available Modules

1. **Line Crossing Detector**
   - Detects when objects cross defined lines or boundaries
   - Useful for entry/exit counting and perimeter monitoring

2. **Visitor Counter**
   - Maintains accurate counts of entries and exits
   - Provides statistics on store traffic and occupancy

3. **Zone Detector**
   - Detects when objects enter or exit defined zones
   - Monitors restricted areas and tracks zone occupancy

4. **Dwell Time Analyzer**
   - Analyzes how long objects stay in specific areas
   - Detects unusual dwell times that might indicate suspicious behavior

5. **Movement Anomaly Detector**
   - Identifies unusual movement patterns
   - Detects erratic behavior and suspicious trajectories

6. **Unauthorized Area Detector**
   - Detects when objects enter areas marked as unauthorized
   - Uses label data to define restricted areas
   - Generates alerts for unauthorized access

7. **Detector Integration**
   - Combines all specialized detectors into a complete system
   - Provides centralized alert handling and visualization

## Using the Modules

Each module can be used independently or together. For detailed usage examples, see the [module documentation](backend/app/utils/README.md).

### Basic Example

```python
from backend.app.utils.detector_integration import DetectorIntegration

# Create integrated detector
detector = DetectorIntegration(
    store_id="my_store",
    labels_file="data/labels.csv"  # For unauthorized area detection
)

# Process object positions (e.g., from a tracking system)
for object_id, position in tracked_objects.items():
    # position should be normalized coordinates (0-1)
    alerts = detector.update(object_id, position)
    
    # Handle any alerts
    for alert in alerts:
        print(f"Alert: {alert}")

# Get current statistics
stats = detector.get_current_stats()
```

## Unauthorized Area Detection

The system can detect when people enter areas marked as unauthorized using label data from a CSV file:

```csv
label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height
UnAuthorized_Area,248,424,468,499,Screenshot.png,1841,969
```

To use this feature:

1. Create a CSV file with labeled unauthorized areas
2. Pass the file path to the detector initialization
3. The system will generate alerts when objects enter these areas

```python
from backend.app.utils.unauthorized_area_detector import UnauthorizedAreaDetector

# Create detector with label data
detector = UnauthorizedAreaDetector(labels_file="data/labels.csv")

# Update with object positions
event = detector.update(object_id=1, position=(0.3, 0.5))
if event:
    print(f"Unauthorized access: {event['message']}")
```

## Interactive Testing

The test script allows you to interactively test the detectors:

1. Run the test script with the `--show` flag
2. Click on the video window to add/move objects
3. Watch how the detectors respond to object movements
4. Press 'q' to exit

## Customization

Each detector module has configurable parameters that can be adjusted for your specific use case. See the individual module documentation for details. 