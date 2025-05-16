import sqlite3
import os
import json
import numpy as np
from datetime import datetime

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Handle dictionaries recursively
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples recursively
        return [convert_to_serializable(x) for x in obj]
    return obj

class FootfallDatabase:
    def __init__(self, db_path='footfall_data.db'):
        """Initialize database for storing footfall data"""
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _get_connection(self):
        """Get a database connection (creating one if needed)"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def _initialize_db(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create footfall summary table (5-minute intervals)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS footfall_summary (
            timestamp TEXT,
            potential_customers INTEGER,
            window_shoppers INTEGER,
            staff INTEGER,
            total_entries INTEGER,
            total_exits INTEGER,
            PRIMARY KEY (timestamp)
        )
        ''')
        
        # Create detailed detections table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            tracking_id TEXT,
            person_type TEXT,
            position TEXT,
            bbox TEXT
        )
        ''')
        
        # Create session table for tracking overall metrics per video session
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            source TEXT,
            total_frames INTEGER,
            avg_fps REAL,
            total_entries INTEGER,
            total_exits INTEGER,
            device_id TEXT
        )
        ''')
        
        # Store analytics events (entry, exit, etc.)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            person_type TEXT,
            tracking_id TEXT,
            details TEXT
        )
        ''')

        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_footfall_timestamp ON footfall_summary(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_tracking_id ON detections(tracking_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        
        conn.commit()
        conn.close()
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def insert_footfall_summary(self, timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits):
        """Insert or update the footfall summary data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        iso_timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        
        cursor.execute('''
        INSERT OR REPLACE INTO footfall_summary 
        (timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (iso_timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits))
        
        conn.commit()
    
    def insert_detection(self, timestamp, frame_number, tracking_id, person_type, position, bbox):
        """Insert a single detection record"""
        conn = self._get_connection()
        cursor = conn.cursor()
        iso_timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        
        # Convert numpy values to Python native types
        position = convert_to_serializable(position)
        bbox = convert_to_serializable(bbox)
        
        # Convert position and bbox to JSON strings using the custom encoder
        position_json = json.dumps(position, cls=NumpyJSONEncoder)
        bbox_json = json.dumps(bbox, cls=NumpyJSONEncoder)
        
        cursor.execute('''
        INSERT INTO detections 
        (timestamp, frame_number, tracking_id, person_type, position, bbox)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (iso_timestamp, frame_number, tracking_id, person_type, position_json, bbox_json))
        
        conn.commit()
    
    def insert_event(self, timestamp, event_type, person_type, tracking_id, details=None):
        """Insert an event record (entry, exit, etc.)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        iso_timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        
        # Convert any numpy values in details recursively
        if details:
            details = convert_to_serializable(details)
        
        # Use custom encoder for JSON serialization
        details_json = json.dumps(details, cls=NumpyJSONEncoder) if details else None
        
        cursor.execute('''
        INSERT INTO events 
        (timestamp, event_type, person_type, tracking_id, details)
        VALUES (?, ?, ?, ?, ?)
        ''', (iso_timestamp, event_type, person_type, tracking_id, details_json))
        
        conn.commit()
    
    def start_session(self, source, device_id=None):
        """Start a new recording session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        start_time = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO sessions 
        (start_time, source, device_id, total_frames, avg_fps, total_entries, total_exits)
        VALUES (?, ?, ?, 0, 0.0, 0, 0)
        ''', (start_time, source, device_id))
        
        conn.commit()
        return cursor.lastrowid
    
    def end_session(self, session_id, total_frames, avg_fps, total_entries, total_exits):
        """Update session with final stats"""
        conn = self._get_connection()
        cursor = conn.cursor()
        end_time = datetime.now().isoformat()
        
        cursor.execute('''
        UPDATE sessions 
        SET end_time = ?, total_frames = ?, avg_fps = ?, total_entries = ?, total_exits = ?
        WHERE id = ?
        ''', (end_time, total_frames, avg_fps, total_entries, total_exits, session_id))
        
        conn.commit()
    
    def get_summary_by_timerange(self, start_time, end_time):
        """Get footfall summary data for a specific time range"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits
        FROM footfall_summary
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        ''', (start_time, end_time))
        
        results = cursor.fetchall()
        return [
            {
                'timestamp': row[0],
                'potential_customers': row[1],
                'window_shoppers': row[2],
                'staff': row[3],
                'total_entries': row[4],
                'total_exits': row[5]
            }
            for row in results
        ]
    
    def get_events_by_timerange(self, start_time, end_time, event_type=None, person_type=None):
        """Get events for a specific time range with optional filtering"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
        SELECT timestamp, event_type, person_type, tracking_id, details
        FROM events
        WHERE timestamp BETWEEN ? AND ?
        '''
        params = [start_time, end_time]
        
        if event_type:
            query += ' AND event_type = ?'
            params.append(event_type)
        
        if person_type:
            query += ' AND person_type = ?'
            params.append(person_type)
        
        query += ' ORDER BY timestamp'
        
        cursor.execute(query, params)
        
        results = cursor.fetchall()
        return [
            {
                'timestamp': row[0],
                'event_type': row[1],
                'person_type': row[2],
                'tracking_id': row[3],
                'details': json.loads(row[4]) if row[4] else None
            }
            for row in results
        ]
    
    def get_latest_stats(self, limit=1):
        """Get the latest footfall statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits
        FROM footfall_summary
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        return [
            {
                'timestamp': row[0],
                'potential_customers': row[1],
                'window_shoppers': row[2],
                'staff': row[3],
                'total_entries': row[4],
                'total_exits': row[5]
            }
            for row in results
        ]
    
    def get_sessions(self, limit=20):
        """Get the most recent sessions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, start_time, end_time, source, total_frames, avg_fps, total_entries, total_exits
        FROM sessions
        ORDER BY start_time DESC
        LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        return [
            {
                'id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'source': row[3],
                'total_frames': row[4],
                'avg_fps': row[5],
                'total_entries': row[6],
                'total_exits': row[7]
            }
            for row in results
        ] 