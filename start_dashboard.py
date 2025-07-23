#!/usr/bin/env python3
"""
Startup script for Claude Manager Service Web Dashboard

This script starts the web dashboard with proper environment setup
and error handling.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables if not already set
os.environ.setdefault('DASHBOARD_HOST', '0.0.0.0')
os.environ.setdefault('DASHBOARD_PORT', '5000')
os.environ.setdefault('FLASK_DEBUG', 'False')

# Import and run the dashboard
try:
    from web.app import app, logger
    
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Get configuration
    host = os.environ.get('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Claude Manager Service Dashboard on http://{host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run the application
    app.run(host=host, port=port, debug=debug)
    
except ImportError as e:
    print(f"Error importing dashboard modules: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting dashboard: {e}")
    sys.exit(1)