"""
Footfall Analytics API Server

This module starts the Flask API server for accessing footfall analytics data.
"""

import os
import argparse
from .api import run_api_server

def parse_args():
    parser = argparse.ArgumentParser(description='Footfall Analytics API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5555, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting Footfall Analytics API server on {args.host}:{args.port}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    run_api_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 