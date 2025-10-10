import sys
import os

# Add scripts folder to path
sys.path.append('scripts')

print("🔍 DETAILED ANALYSIS OF YOUR SCRIPTS FOLDER")

# List all files in scripts folder
print("\n📁 Files in scripts folder:")
for file in os.listdir('scripts'):
    if file.endswith('.py'):
        print(f"  - {file}")

# Test each file individually
print("\n🧪 Testing individual imports:")

# Test translator.py in detail
try:
    import translator
    print(f"✅ translator.py imported successfully!")
    print(f"   Available in translator: {[x for x in dir(translator) if not x.startswith('_')]}")
    
    # Check for common function names
    common_functions = ['main', 'process_frame', 'detect', 'predict', 'real_time_detection', 'translate']
    for func in common_functions:
        if hasattr(translator, func):
            print(f"   🎯 Found: translator.{func}")
            
except Exception as e:
    print(f"❌ translator.py import failed: {e}")

# Test landmark_extraction.py in detail
try:
    import landmark_extraction
    print(f"✅ landmark_extraction.py imported successfully!")
    print(f"   Available in landmark_extraction: {[x for x in dir(landmark_extraction) if not x.startswith('_')]}")
    
    common_functions = ['extract_landmarks', 'process_landmarks', 'detect_hands']
    for func in common_functions:
        if hasattr(landmark_extraction, func):
            print(f"   🎯 Found: landmark_extraction.{func}")
            
except Exception as e:
    print(f"❌ landmark_extraction.py import failed: {e}")

# Test if there's a main execution pattern
print("\n🔧 Checking execution patterns:")
try:
    import translator
    # Check if we can call something
    if hasattr(translator, 'main'):
        print("   translator has main() - might be standalone script")
    if hasattr(translator, 'predict'):
        print("   translator has predict() - good for integration")
except:
    pass
