# Specialized Detector Modules

This directory contains specialized detector modules, each with a single responsibility. These modules are designed to be used individually or together to build a complete detection system.

## Design Philosophy

Each module follows the Single Responsibility Principle:
- Each class has a single, well-defined responsibility
- Modules are focused on doing one thing well
- Easy to test, maintain, and extend
- Can be used independently or combined

## Available Modules

### 1. Line Crossing Detector (`line_crossing_detector.py`)

Detects when objects cross defined lines or boundaries. Useful for:
- Entry/exit counting
- Perimeter monitoring
- Boundary violation detection

```python
# Example usage
from line_crossing_detector import LineCrossingDetector

# Create a line crossing detector
detector = LineCrossingDetector(
    line_start=(0.0, 0.5),  # Normalized coordinates (0-1)
    line_end=(0.2, 0.5),
    line_id="entry_line",
    crossing_direction="positive"  # Count only entries
)

# Update with object positions
event = detector.update(object_id=1, position=(0.1, 0.6))
if event:
    print(f"Line crossed: {event}")
```

### 2. Visitor Counter (`visitor_counter.py`)

Maintains accurate counts of entries and exits. Useful for:
- Store traffic analysis
- Occupancy monitoring
- Footfall statistics

```python
# Example usage
from visitor_counter import VisitorCounter

# Create a visitor counter
counter = VisitorCounter(
    store_id="store_123",
    data_dir="data/footfall"
)

# Record entries and exits
counter.record_entry()
counter.record_exit()

# Get current stats
stats = counter.get_current_stats()
print(f"Current visitors: {stats['current_visitors']}")
```

### 3. Zone Detector (`zone_detector.py`)

Detects when objects enter or exit defined zones. Useful for:
- Restricted area monitoring
- Zone occupancy tracking
- Activity analysis by area

```python
# Example usage
import numpy as np
from zone_detector import ZoneDetector

# Define zones as polygons
zones = {
    "cash_counter": np.array([
        [0.4, 0.3], [0.6, 0.3], [0.6, 0.4], [0.4, 0.4]
    ]),
    "entrance": np.array([
        [0.0, 0.4], [0.2, 0.4], [0.2, 0.6], [0.0, 0.6]
    ])
}

# Create a zone detector
detector = ZoneDetector(zones=zones)

# Update with object positions
event = detector.update(object_id=1, position=(0.5, 0.35))
if event:
    print(f"Zone event: {event}")

# Check if an object is in a specific zone
is_in_zone = detector.is_object_in_zone(object_id=1, zone_id="cash_counter")
```

### 4. Dwell Time Analyzer (`dwell_time_analyzer.py`)

Analyzes how long objects stay in specific areas. Useful for:
- Customer engagement analysis
- Suspicious behavior detection
- Service time monitoring

```python
# Example usage
import numpy as np
from dwell_time_analyzer import DwellTimeAnalyzer

# Define areas to monitor
areas = {
    "cash_counter": np.array([
        [0.4, 0.3], [0.6, 0.3], [0.6, 0.4], [0.4, 0.4]
    ])
}

# Define thresholds for each area
thresholds = {
    "cash_counter": {
        "short": 5.0,    # 5 seconds is unusually short
        "normal": 60.0,  # 1 minute is normal
        "long": 180.0    # 3 minutes is unusually long
    }
}

# Create a dwell time analyzer
analyzer = DwellTimeAnalyzer(areas=areas, thresholds=thresholds)

# Update with object positions
alert = analyzer.update(object_id=1, position=(0.5, 0.35))
if alert:
    print(f"Dwell time alert: {alert}")
```

### 5. Movement Anomaly Detector (`movement_anomaly_detector.py`)

Detects unusual movement patterns. Useful for:
- Suspicious behavior detection
- Erratic movement identification
- Theft pattern recognition

```python
# Example usage
from movement_anomaly_detector import MovementAnomalyDetector

# Create a movement anomaly detector
detector = MovementAnomalyDetector(
    suspicious_threshold=0.7  # Higher values require more suspicious behavior
)

# Update with object positions (call regularly with new positions)
alert = detector.update(object_id=1, position=(0.5, 0.35))
if alert:
    print(f"Movement anomaly alert: {alert}")
```

### 6. Detector Integration (`detector_integration.py`)

Integrates all specialized detectors into a complete system. Useful for:
- Building a full-featured detection system
- Combining multiple detection capabilities
- Centralized alert handling

```python
# Example usage
from detector_integration import DetectorIntegration

# Create an integrated detector system
integration = DetectorIntegration(
    store_id="store_123",
    data_dir="data"
)

# Update with object positions
alerts = integration.update(object_id=1, position=(0.5, 0.35))
if alerts:
    for alert in alerts:
        print(f"Alert: {alert}")

# Get current statistics
stats = integration.get_current_stats()
```

## Testing the Modules

Use the `test_specialized_detectors.py` script in the `hackathon/tools` directory to test the modules:

```bash
# Test line crossing detector with the sample video
python hackathon/tools/test_specialized_detectors.py --mode line --source hackathon/data/sample_video.mp4 --show

# Test zone detector with the sample video
python hackathon/tools/test_specialized_detectors.py --mode zone --source hackathon/data/sample_video.mp4 --show

# Test the integrated system with the sample video
python hackathon/tools/test_specialized_detectors.py --mode integration --source hackathon/data/sample_video.mp4 --show
```

### Generating a Sample Video

You can generate a new sample video for testing using the provided script:

```bash
# Generate a sample video
python hackathon/tools/generate_sample_video.py
```

This will create a video file at `hackathon/data/sample_video.mp4` with moving objects that interact with the detection zones and lines.

## Design Benefits

1. **Modularity**: Each component can be used independently
2. **Testability**: Easy to test individual components
3. **Maintainability**: Simple, focused code is easier to maintain
4. **Extensibility**: Easy to add new features or modify existing ones
5. **Reusability**: Components can be reused in different contexts 