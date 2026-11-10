#!/usr/bin/env python3
"""
Convenience entry point for scripts/verify_all.py
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "verify_all.py")
    sys.exit(subprocess.call([sys.executable, script_path] + sys.argv[1:]))
