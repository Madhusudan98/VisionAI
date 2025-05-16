import os
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from datetime import datetime, timedelta
import sqlite3
import json

from .database import FootfallDatabase

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database path - use absolute path to the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DB_PATH = os.path.join(project_root, 'footfall_data.db')

def get_db():
    """Get database connection for the current request"""
    if 'db' not in g:
        g.db = FootfallDatabase(DB_PATH)
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """Close database connection when the request ends"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# API Routes
@app.route('/api/status', methods=['GET'])
def get_status():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'db_path': DB_PATH  # Add database path to status to help with debugging
    })

@app.route('/api/stats/latest', methods=['GET'])
def get_latest_stats():
    """Get the latest footfall statistics"""
    try:
        db = get_db()
        limit = int(request.args.get('limit', 1))
        stats = db.get_latest_stats(limit)
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/all', methods=['GET'])
def get_all_stats():
    """Get all footfall statistics"""
    try:
        db = get_db()
        # Get query parameters with defaults
        conn = db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT timestamp, potential_customers, window_shoppers, staff, total_entries, total_exits
        FROM footfall_summary
        ORDER BY timestamp
        ''')
        
        results = cursor.fetchall()
        stats = [
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
        
        return jsonify({
            'success': True,
            'count': len(stats),
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/timerange', methods=['GET'])
def get_stats_by_timerange():
    """Get footfall statistics for a specific time range"""
    try:
        db = get_db()
        # Get query parameters with defaults
        now = datetime.now()
        end_time = request.args.get('end_time', now.isoformat())
        
        # Default to last hour if start_time not specified
        default_start = (now - timedelta(hours=1)).isoformat()
        start_time = request.args.get('start_time', default_start)
        
        stats = db.get_summary_by_timerange(start_time, end_time)
        return jsonify({
            'success': True,
            'data': stats,
            'timerange': {
                'start': start_time,
                'end': end_time
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get events for a specific time range with optional filtering"""
    try:
        db = get_db()
        # Get query parameters with defaults
        now = datetime.now()
        end_time = request.args.get('end_time', now.isoformat())
        
        # Default to last hour if start_time not specified
        default_start = (now - timedelta(hours=1)).isoformat()
        start_time = request.args.get('start_time', default_start)
        
        # Optional filters
        event_type = request.args.get('event_type')
        person_type = request.args.get('person_type')
        
        events = db.get_events_by_timerange(start_time, end_time, event_type, person_type)
        return jsonify({
            'success': True,
            'data': events,
            'timerange': {
                'start': start_time,
                'end': end_time
            },
            'filters': {
                'event_type': event_type,
                'person_type': person_type
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get recent recording sessions"""
    try:
        db = get_db()
        limit = int(request.args.get('limit', 20))
        sessions = db.get_sessions(limit)
        return jsonify({
            'success': True,
            'data': sessions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get a summary of all analytics data"""
    try:
        db = get_db()
        # Get the last session
        sessions = db.get_sessions(1)
        latest_session = sessions[0] if sessions else None
        
        # Get the latest stats
        latest_stats = db.get_latest_stats(1)
        latest = latest_stats[0] if latest_stats else None
        
        # Get stats for today
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day).isoformat()
        today_stats = db.get_summary_by_timerange(today_start, now.isoformat())
        
        # Calculate today's totals
        today_entries = sum(stat['total_entries'] for stat in today_stats) if today_stats else 0
        today_exits = sum(stat['total_exits'] for stat in today_stats) if today_stats else 0
        
        return jsonify({
            'success': True,
            'latest': latest,
            'today': {
                'entries': today_entries,
                'exits': today_exits,
                'date': today_start
            },
            'latest_session': latest_session
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/events/all', methods=['GET'])
def get_all_events():
    """Get all events"""
    try:
        db = get_db()
        conn = db._get_connection()
        cursor = conn.cursor()
        
        # Optional type filters
        event_type = request.args.get('event_type')
        person_type = request.args.get('person_type')
        
        query = '''
        SELECT timestamp, event_type, person_type, tracking_id, details
        FROM events
        '''
        params = []
        
        where_clauses = []
        if event_type:
            where_clauses.append('event_type = ?')
            params.append(event_type)
        
        if person_type:
            where_clauses.append('person_type = ?')
            params.append(person_type)
        
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        query += ' ORDER BY timestamp'
        
        cursor.execute(query, params)
        
        results = cursor.fetchall()
        events = [
            {
                'timestamp': row[0],
                'event_type': row[1],
                'person_type': row[2],
                'tracking_id': row[3],
                'details': json.loads(row[4]) if row[4] else None
            }
            for row in results
        ]
        
        return jsonify({
            'success': True,
            'count': len(events),
            'data': events
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def run_api_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_api_server(debug=True) 