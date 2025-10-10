import sys
import os

# Add scripts folder to path
sys.path.append('scripts')

print("🔍 Checking what's available in your scripts folder...")

# Test translator.py
try:
    from translator import *
    print("✅ translator.py imported successfully!")
    print("Functions in translator.py:", [f for f in dir() if not f.startswith('_')])
except Exception as e:
    print(f"❌ Could not import translator.py: {e}")

# Test landmark_extraction.py  
try:
    from landmark_extraction import *
    print("✅ landmark_extraction.py imported successfully!")
    print("Functions in landmark_extraction.py:", [f for f in dir() if not f.startswith('_')])
except Exception as e:
    print(f"❌ Could not import landmark_extraction.py: {e}")

# Check if there's a main function
try:
    import translator
    if hasattr(translator, 'main'):
        print("🎯 translator.py has 'main' function")
    if hasattr(translator, 'real_time_detection'):
        print("🎯 translator.py has 'real_time_detection' function")
    if hasattr(translator, 'process_frame'):
        print("🎯 translator.py has 'process_frame' function")
    if hasattr(translator, 'predict'):
        print("🎯 translator.py has 'predict' function")
except Exception as e:
    print(f"❌ Error examining translator: {e}")