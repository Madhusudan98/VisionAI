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
│           ├── detector_integration.py     # Integrates all detector modules
│           └── README.md                   # Detailed documentation for each module
├── data/
│   └── sample_video.mp4                    # Sample video for testing
└── tools/
    ├── test_specialized_detectors.py       # Script to test the detector modules
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

6. **Detector Integration**
   - Combines all specialized detectors into a complete system
   - Provides centralized alert handling and visualization

## Using the Modules

Each module can be used independently or together. For detailed usage examples, see the [module documentation](backend/app/utils/README.md).

### Basic Example

```python
from backend.app.utils.detector_integration import DetectorIntegration

# Create integrated detector
detector = DetectorIntegration(store_id="my_store")

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

## Interactive Testing

The test script allows you to interactively test the detectors:

1. Run the test script with the `--show` flag
2. Click on the video window to add/move objects
3. Watch how the detectors respond to object movements
4. Press 'q' to exit

## Customization

Each detector module has configurable parameters that can be adjusted for your specific use case. See the individual module documentation for details. 