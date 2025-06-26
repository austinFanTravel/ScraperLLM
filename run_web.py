#!/usr/bin/env python3
"""
Run the ScraperLLM web interface
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scraper_llm.web.app import run_server

import argparse
import uvicorn
from pathlib import Path
import sys

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    # Add the project root to the Python path
    project_root = str(Path(__file__).parent.absolute())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Starting ScraperLLM web interface...")
    print(f"Access the web interface at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "scraper_llm.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=False,
        reload_dirs=[str(Path(__file__).parent / "scraper_llm")] if reload else None
    )

def main():
    """Parse command line arguments and run the server"""
    parser = argparse.ArgumentParser(description="Run ScraperLLM web interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload")
    parser.set_defaults(reload=False)
    
    args = parser.parse_args()
    
    try:
        run_server(host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
