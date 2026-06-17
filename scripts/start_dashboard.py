#!/usr/bin/env python
"""
Start the Smart Knowledge Hub Dashboard
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Start the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent.parent / "src" / "observability" / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard app not found at {dashboard_path}")
        sys.exit(1)

    print(f"Starting Smart Knowledge Hub Dashboard...")
    print(f"Opening at http://localhost:8501")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            cwd=str(Path(__file__).parent.parent)
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
