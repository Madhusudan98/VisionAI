import cv2
import numpy as np
import sys
import os
import time
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.utils.object_detector import ObjectDetector
from backend.app.database import FootfallDatabase

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source (file path, URL, or webcam index)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--db-path', type=str, default='footfall_data.db', help='Path to database file')
    parser.add_argument('--device-id', type=str, default=None, help='Device ID for camera identification')
    parser.add_argument('--log-interval', type=int, default=30, help='Interval in seconds to log summary data')
    return parser.parse_args()

def draw_stats(frame, potential, window_shoppers, staff, entries, exits):
    """Display footfall statistics on the frame in bottom right with smaller font"""
    h, w = frame.shape[:2]
    
    # Calculate overlay dimensions based on frame size
    overlay_width = 240
    overlay_height = 160
    
    # Position in bottom right
    x_pos = w - overlay_width - 10
    y_pos = h - overlay_height - 10
    
    # Draw transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + overlay_width, y_pos + overlay_height), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Smaller font size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_thickness = 1
    line_height = 25
    
    # Starting position for text
    text_x = x_pos + 10
    text_y = y_pos + 25
    
    # Draw stats with smaller font
    cv2.putText(frame, f"Potential Customers: {potential}", (text_x, text_y), 
                font, font_size, (0, 255, 255), font_thickness)
    cv2.putText(frame, f"Window Shoppers: {window_shoppers}", (text_x, text_y + line_height), 
                font, font_size, (255, 165, 0), font_thickness)
    cv2.putText(frame, f"Cashiers/Staff: {staff}", (text_x, text_y + 2*line_height),
                font, font_size, (255, 0, 255), font_thickness)
    cv2.putText(frame, f"Total Entries: {entries}", (text_x, text_y + 3*line_height), 
                font, font_size, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Total Exits: {exits}", (text_x, text_y + 4*line_height), 
                font, font_size, (0, 0, 255), font_thickness)
    
    return frame

# Globals for mouse callback
clicking = False
points = []
current_frame = None
cash_counter_zone = None
shopping_zone = None

def click_and_crop(event, x, y, flags, param):
    """Mouse callback function for selecting areas"""
    global clicking, points, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicking = True
        points.append((x, y))
        
        # Draw a point where the user clicked
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Cash Counter (4 points)", current_frame)
    
    elif event == cv2.EVENT_LBUTTONUP:
        clicking = False

def main():
    global current_frame, cash_counter_zone, shopping_zone, points
    args = parse_args()
    
    # Create output directory if saving results
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize database
    db = FootfallDatabase(args.db_path)
    
    # Initialize detector
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.conf_thres,
        classes=[0]  # Only detect people (class 0)
    )
    
    # Initialize video capture
    try:
        source = args.source
        if source.isnumeric():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the first frame for manual zone selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return
    
    # Make a copy for display
    current_frame = first_frame.copy()
    
    # Set up mouse callback for manual selection
    cv2.namedWindow("Select Cash Counter (4 points)")
    cv2.setMouseCallback("Select Cash Counter (4 points)", click_and_crop)
    
    # Instruction text
    cv2.putText(current_frame, "Click 4 points to define cash counter area, then press ENTER", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show frame and wait for input
    cv2.imshow("Select Cash Counter (4 points)", current_frame)
    
    # Wait for 4 points or Enter key
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
    
    # Close selection window
    cv2.destroyWindow("Select Cash Counter (4 points)")
    
    # Define cash counter zone from selected points or use default
    if len(points) == 4:
        cash_counter_zone = np.array(points, np.int32)
        print(f"Cash counter zone set to: {points}")
    else:
        # Default cash counter if not enough points selected
        cash_counter_zone = np.array([
            [int(0.5 * width), int(0.2 * height)],
            [int(0.9 * width), int(0.2 * height)],
            [int(0.9 * width), int(0.5 * height)],
            [int(0.5 * width), int(0.5 * height)]
        ], np.int32)
        print("Using default cash counter zone")
    
    # Define shopping zone as the rest of the frame
    # For now, set a general shopping zone that excludes the cash counter area
    shopping_zone = np.array([
        [int(0.1 * width), int(0.1 * height)],
        [int(0.9 * width), int(0.1 * height)],
        [int(0.9 * width), int(0.9 * height)],
        [int(0.1 * width), int(0.9 * height)]
    ], np.int32)
    
    # Reset the cap to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize video writer if saving
    out = None
    if args.save:
        output_path = os.path.join(args.output_dir, f"output_{int(time.time())}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Initialize counters
    total_entries = 0
    total_exits = 0
    potential_customers = 0
    window_shoppers = 0
    staff_count = 0
    
    # Initialize tracking state
    tracked_people = {}  # track_id -> data
    
    # First frame flag
    first_frame = True
    
    # Start a new session in the database
    session_id = db.start_session(source=args.source, device_id=args.device_id)
    
    # Initialize timestamp for periodic logging
    last_log_time = time.time()
    current_timestamp = datetime.now()
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    # Function to check if point is in polygon - FIXED FOR CORRECT TYPE HANDLING
    def is_in_polygon(point, polygon):
        # Convert point to tuple of floats for OpenCV
        point = (float(point[0]), float(point[1]))
        
        # Reshape polygon to the correct format and convert to int32
        polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
        
        # Use the OpenCV function
        try:
            return cv2.pointPolygonTest(polygon, point, False) >= 0
        except cv2.error:
            # Fallback to a simpler method if pointPolygonTest fails
            # Create a bounding box
            x_coords = [p[0][0] for p in polygon]
            y_coords = [p[0][1] for p in polygon]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # Check if the point is within the bounding box
            return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax
        
    # Function to calculate distance from a point to a polygon
    def distance_to_polygon(point, polygon):
        # Convert point to tuple of floats for OpenCV
        point = (float(point[0]), float(point[1]))
        
        # Convert polygon to the format OpenCV expects
        polygon_arr = np.array(polygon, dtype=np.int32)
        
        # Debug
        print(f"Point: {point}, Polygon shape: {polygon_arr.shape}")
        
        # Check if person is inside polygon first
        is_inside = is_in_polygon(point, polygon_arr)
        if is_inside:
            return 0  # No distance if inside
        
        # Calculate minimum distance to any edge of the polygon
        min_dist = float('inf')
        
        # Iterate through each edge of the polygon
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i+1) % len(polygon)]
            
            # Calculate distance to the line segment
            line_len_sq = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
            if line_len_sq == 0:  # Handle zero-length edge
                dist = np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
            else:
                # Calculate projection onto line segment
                t = max(0, min(1, ((point[0] - p1[0]) * (p2[0] - p1[0]) + 
                                   (point[1] - p1[1]) * (p2[1] - p1[1])) / line_len_sq))
                proj_x = p1[0] + t * (p2[0] - p1[0])
                proj_y = p1[1] + t * (p2[1] - p1[1])
                dist = np.sqrt((point[0] - proj_x)**2 + (point[1] - proj_y)**2)
            
            min_dist = min(min_dist, dist)
            
        return min_dist
    
    # Define proximity threshold (in pixels) - approximates a 1-meter range
    # This can be adjusted based on the video's scale and perspective
    proximity_threshold = int(0.2 * max(width, height))  # 2% of max dimension for a stricter threshold
    print(f"Proximity threshold set to {proximity_threshold} pixels")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        current_timestamp = datetime.now()
        
        # Get detections
        detections = detector.detect(frame)
        
        # Current frame person IDs
        current_ids = set()
        
        # Count people visible in the current frame by type
        potential_count = 0
        window_shopper_count = 0
        staff_count_current = 0
        
        # Draw zones
        cv2.polylines(frame, [shopping_zone], True, (0, 255, 255), 2)
        cv2.putText(frame, "Shopping Zone", (shopping_zone[0][0] + 10, shopping_zone[0][1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.polylines(frame, [cash_counter_zone], True, (255, 0, 255), 2)
        cv2.putText(frame, "Cash Counter", (cash_counter_zone[0][0] + 10, cash_counter_zone[0][1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Process each detected person
        for obj in detections["objects"]:
            if obj["class_id"] == 0:  # Person class
                x1, y1, x2, y2 = obj["bbox"]
                tracking_id = obj["tracking_id"]
                current_ids.add(tracking_id)
                
                # Calculate center position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                position = (center_x, center_y)
                
                # Check which zone the person is in
                in_cash_counter = is_in_polygon(position, cash_counter_zone)
                in_shopping = is_in_polygon(position, shopping_zone)
                
                # Calculate distance to cash counter if not already in it
                near_cash_counter = False
                if not in_cash_counter:
                    distance = distance_to_polygon(position, cash_counter_zone)
                    print(f"Distance to cash counter: {distance}")
                    near_cash_counter = distance <= proximity_threshold
                    # Debug info
                
                # Initialize or update tracking data
                if tracking_id not in tracked_people:
                    # New person - start everyone as window shopper by default
                    tracked_people[tracking_id] = {
                        "first_seen": frame_count,
                        "last_seen": frame_count,
                        "in_shopping": in_shopping,
                        "in_cash_counter": in_cash_counter,
                        "position": position,
                        "frames_in_shopping": 1 if in_shopping else 0,
                        "frames_in_cash_counter": 1 if in_cash_counter else 0,
                        "total_frames": 1,
                        "last_cash_counter_visit": 0,  # Start with 0 to indicate never visited
                        "ever_near_cash_counter": False,  # New flag to track if ever near cash counter
                        "type": "window_shopper",  # Start as window shopper by default
                        "counted": first_frame  # Don't count people in first frame as new entries
                    }
                else:
                    # Update existing person
                    person = tracked_people[tracking_id]
                    person["last_seen"] = frame_count
                    person["position"] = position
                    person["total_frames"] += 1
                    
                    # Update zone counters
                    if in_shopping:
                        person["frames_in_shopping"] += 1
                    if in_cash_counter:
                        person["frames_in_cash_counter"] += 1
                        person["last_cash_counter_visit"] = frame_count  # Update timing of last visit
                    elif near_cash_counter:
                        person["ever_near_cash_counter"] = True
                    
                    # Update zone status
                    person["in_shopping"] = in_shopping
                    person["in_cash_counter"] = in_cash_counter
                
                # Classify person type based on zone behavior
                person = tracked_people[tracking_id]
                
                # Default values - ensure these are always set
                color = (255, 255, 255)  # White (default)
                label = f"UNKNOWN {tracking_id}"  # Default label
                
                # Calculate ratios safely
                shopping_ratio = person["frames_in_shopping"] / max(1, person["total_frames"])
                cash_counter_ratio = person["frames_in_cash_counter"] / max(1, person["total_frames"])
                
                # Define "recency window" - how many frames back is considered "recent"
                recency_window = 30  # Consider cash counter visits in last 30 frames as "recent"
                
                # Only count as recently visited if they've actually been to the cash counter
                recently_visited_cash = False
                if "last_cash_counter_visit" in person and person["last_cash_counter_visit"] > 0:
                    recently_visited_cash = (frame_count - person["last_cash_counter_visit"]) < recency_window
                
                # Add debug output for the first few people
                if frame_count % 30 == 0 and tracking_id < 5:
                    print(f"Person {tracking_id}: in_cash={in_cash_counter}, near={near_cash_counter}, recent={recently_visited_cash}, ever_near={person['ever_near_cash_counter']}")
                
                # SIMPLIFIED CLASSIFICATION LOGIC:
                # 1. People who spend a lot of time at cash counter are staff
                if cash_counter_ratio > 0.3 and person["frames_in_cash_counter"] > 15:
                    person["type"] = "staff" 
                    staff_count_current += 1
                    color = (255, 0, 255)  # Purple
                    label = f"STAFF {tracking_id}"
                # 2. People only at or VERY near the cash counter are potential customers
                # Removed "recently_visited_cash" to make classification more strict
                elif in_cash_counter or near_cash_counter:
                    person["type"] = "potential"
                    potential_count += 1
                    color = (0, 255, 255)  # Cyan
                    label = f"POTENTIAL {tracking_id}"
                # 3. Everyone else is a window shopper
                else:
                    person["type"] = "window_shopper"
                    window_shopper_count += 1
                    color = (255, 165, 0)  # Orange
                    label = f"WINDOW {tracking_id}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-20), (x1 + text_width + 5, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Count as new entry if this person hasn't been counted yet and isn't staff
                if not person["counted"] and person["total_frames"] >= 10 and person["type"] != "staff":
                    person["counted"] = True
                    total_entries += 1
                    
                    # Log the entry event to the database
                    db.insert_event(
                        timestamp=current_timestamp,
                        event_type="entry",
                        person_type=person["type"],
                        tracking_id=tracking_id,
                        details={
                            "position": position,
                            "frame": frame_count
                        }
                    )
                    
                    if person["type"] == "potential":
                        potential_customers += 1
                    else:
                        window_shoppers += 1
                
                # Store detection in database (only every 10th frame to avoid too much data)
                if frame_count % 10 == 0:
                    db.insert_detection(
                        timestamp=current_timestamp,
                        frame_number=frame_count,
                        tracking_id=tracking_id,
                        person_type=person["type"],
                        position=position,
                        bbox=[x1, y1, x2, y2]
                    )
        
        # Handle exits - people who were tracked but are no longer visible
        for track_id in list(tracked_people.keys()):
            if track_id not in current_ids:
                person = tracked_people[track_id]
                frames_gone = frame_count - person["last_seen"]
                
                # If gone for 10+ frames and was counted as an entry, count as exit
                if frames_gone >= 10 and person["counted"] and person["type"] != "staff":
                    total_exits += 1
                    
                    # Log the exit event to the database
                    db.insert_event(
                        timestamp=current_timestamp,
                        event_type="exit",
                        person_type=person["type"],
                        tracking_id=track_id,
                        details={
                            "frames_visible": person["total_frames"],
                            "first_seen": person["first_seen"],
                            "last_seen": person["last_seen"]
                        }
                    )
                    
                    if person["type"] == "potential":
                        potential_customers = max(0, potential_customers - 1)
                    else:
                        window_shoppers = max(0, window_shoppers - 1)
                    
                    # Remove from tracking
                    del tracked_people[track_id]
                # Remove very stale tracks even if they weren't counted
                elif frames_gone >= 30:
                    del tracked_people[track_id]
        
        # After first frame, update flag
        if first_frame:
            first_frame = False
            
            # Initialize staff count
            staff_count = staff_count_current
            
            # Consider all initial people (except staff) as entries
            for person in tracked_people.values():
                if person["type"] != "staff":
                    total_entries += 1
                    if person["type"] == "potential":
                        potential_customers += 1
                    else:
                        window_shoppers += 1
        else:
            # Update staff count only when it changes
            if staff_count_current != staff_count:
                staff_count = staff_count_current
        
        # Draw stats in bottom right with smaller font
        frame = draw_stats(frame, potential_customers, window_shoppers, staff_count, total_entries, total_exits)
        
        # Draw proximity threshold for visualization
        if args.show:  # Only draw when display is enabled to avoid unnecessary computation
            # Draw the proximity outline around cash counter
            # Create a slightly expanded polygon
            expanded_cash_counter = []
            for point in cash_counter_zone:
                expanded_cash_counter.append(point)
            
            # Convert to proper format
            expanded_cash_counter = np.array(expanded_cash_counter, np.int32).reshape((-1, 1, 2))
            # Draw dashed line for the proximity threshold
            dash_length = 10
            for i in range(len(cash_counter_zone)):
                p1 = cash_counter_zone[i]
                p2 = cash_counter_zone[(i+1) % len(cash_counter_zone)]
                # Draw dashed line
                cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 255), 1)
            
            # Display the proximity threshold text
            cv2.putText(frame, f"Proximity threshold: ~1m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add processing information (smaller and also in bottom left)
        elapsed = current_time - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add FPS counter at bottom left with smaller font
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Periodically log summary data to database
        if current_time - last_log_time >= args.log_interval:
            # Log current stats to database
            db.insert_footfall_summary(
                timestamp=current_timestamp,
                potential_customers=potential_customers,
                window_shoppers=window_shoppers,
                staff=staff_count,
                total_entries=total_entries,
                total_exits=total_exits
            )
            last_log_time = current_time
            print(f"Logged data at {current_timestamp}: Potential={potential_customers}, Window={window_shoppers}, Staff={staff_count}")
        
        # Show results
        if args.show:
            cv2.imshow("Customer Analytics", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Save results
        if args.save and out:
            out.write(frame)
    
    # Update session data in the database
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    db.end_session(
        session_id=session_id,
        total_frames=frame_count,
        avg_fps=avg_fps,
        total_entries=total_entries,
        total_exits=total_exits
    )
    
    # Final logging of stats
    db.insert_footfall_summary(
        timestamp=datetime.now(),
        potential_customers=potential_customers,
        window_shoppers=window_shoppers,
        staff=staff_count,
        total_entries=total_entries,
        total_exits=total_exits
    )
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    db.close()
    
    # Print summary
    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} FPS)")
    print(f"Final stats: Entries={total_entries}, Exits={total_exits}")
    print(f"Customer types: Potential={potential_customers}, Window Shoppers={window_shoppers}, Staff={staff_count}")
    print(f"Data saved to database: {args.db_path}")

if __name__ == "__main__":
    main() 