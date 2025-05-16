#!/usr/bin/env python
"""
Footfall Analytics API Server Runner

This script starts the Footfall Analytics API server.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.server import main

if __name__ == "__main__":
    main() 