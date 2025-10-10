# test_methods.py
import sys
sys.path.append('scripts')

from translator import SignLanguageTranslator

# Create instance
translator = SignLanguageTranslator()

# Check available methods
print("📋 Available methods in your translator:")
for method_name in dir(translator):
    if not method_name.startswith('_'):  # Skip private methods
        print(f"  - {method_name}")

# Check if it has specific common methods
common_methods = ['translate', 'process_frame', 'predict', 'detect']
for method in common_methods:
    if hasattr(translator, method):
        print(f"✅ Has '{method}' method!")