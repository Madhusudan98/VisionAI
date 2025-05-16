# Footfall Analytics API

This package provides a database for storing footfall detection results and a REST API for accessing the data.

## Features

- SQLite database for time series footfall data
- REST API to query and visualize the data
- Integration with the footfall detector tool

## Prerequisites

Install the required packages:

```bash
pip install -r ../requirements.txt
```

## Running the Footfall Detector with Database Support

To run the footfall detector with database storage:

```bash
python ../tools/footfall-detector.py --source <video_source> --show --db-path <database_path>
```

Common options:
- `--source`: Video source (file path, URL, or webcam index)
- `--db-path`: Path to database file (default: footfall_data.db)
- `--show`: Show detection results
- `--save`: Save video output
- `--device-id`: Device/camera identifier
- `--log-interval`: Interval in seconds to log summary data (default: 30s)

## Running the API Server

To run the API server:

```bash
python ../tools/run_api_server.py --port 5000
```

API server options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 5000)
- `--debug`: Enable debug mode

## API Endpoints

### Health Check
- `GET /api/status`: Check if the API is running

### Statistics
- `GET /api/stats/latest?limit=1`: Get the latest footfall statistics
- `GET /api/stats/timerange?start_time=2023-05-01T00:00:00&end_time=2023-05-02T00:00:00`: Get footfall statistics for a specific time range

### Events
- `GET /api/events?start_time=2023-05-01T00:00:00&end_time=2023-05-02T00:00:00`: Get events (entries, exits) for a specific time range
- `GET /api/events?event_type=entry&person_type=potential`: Filter events by type and person category

### Sessions
- `GET /api/sessions?limit=20`: Get recent recording sessions

### Analytics
- `GET /api/analytics/summary`: Get a summary of all analytics data

## Database Schema

### Tables

1. **footfall_summary**: Summary statistics at regular intervals
   - timestamp
   - potential_customers
   - window_shoppers
   - staff
   - total_entries
   - total_exits

2. **detections**: Detailed detection records
   - id
   - timestamp
   - frame_number
   - tracking_id
   - person_type
   - position
   - bbox

3. **sessions**: Recording session information
   - id
   - start_time
   - end_time
   - source
   - total_frames
   - avg_fps
   - total_entries
   - total_exits
   - device_id

4. **events**: Entry/exit events
   - id
   - timestamp
   - event_type
   - person_type
   - tracking_id
   - details

## API Query Examples

### Get latest statistics
```
curl http://localhost:5000/api/stats/latest
```

### Get statistics for a specific time range
```
curl "http://localhost:5000/api/stats/timerange?start_time=2023-05-01T00:00:00&end_time=2023-05-02T00:00:00"
```

### Get entry events for potential customers
```
curl "http://localhost:5000/api/events?event_type=entry&person_type=potential"
```

### Get a summary of all analytics data
```
curl http://localhost:5000/api/analytics/summary
``` 