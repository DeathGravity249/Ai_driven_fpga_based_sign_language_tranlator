# Run this in Python to see what's available in your scripts
import sys
import os
sys.path.append('scripts')

# Try to import and see what functions are available
try:
    from translator import *
    print("✅ translator.py imported successfully!")
    print("Available functions:", [f for f in dir() if not f.startswith('_')])
except Exception as e:
    print(f"❌ Could not import translator.py: {e}")

try:
    from landmark_extraction import *
    print("✅ landmark_extraction.py imported successfully!")
    print("Available functions:", [f for f in dir() if not f.startswith('_')])
except Exception as e:
    print(f"❌ Could not import landmark_extraction.py: {e}")