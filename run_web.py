"""Start the ThunderSTORM Python web app on http://127.0.0.1:8210 (local only).

Usage: python run_web.py [--port 8210] [--reload]
The repository folder is the thunderstorm_python package, so its parent folder is put on sys.path.
"""
import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT.parent))
    uvicorn.run(f"{ROOT.name}.web.app:app", host="127.0.0.1", port=a.port, reload=a.reload,
                reload_dirs=[str(ROOT)] if a.reload else None)


if __name__ == "__main__":
    main()
