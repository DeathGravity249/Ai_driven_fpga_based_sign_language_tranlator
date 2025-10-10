# test_translator_methods.py
import sys
sys.path.append('scripts')

try:
    from translator import SignLanguageTranslator
    
    # Create instance
    translator = SignLanguageTranslator()
    
    print("📋 Methods available in your translator:")
    for method_name in dir(translator):
        if not method_name.startswith('_'):  # Skip private methods
            print(f"  - {method_name}")
            
    # Test common methods
    common_methods = ['translate', 'process_frame', 'predict', 'detect']
    for method in common_methods:
        if hasattr(translator, method):
            print(f"✅ Has '{method}' method")
            
except Exception as e:
    print(f"❌ Error: {e}"