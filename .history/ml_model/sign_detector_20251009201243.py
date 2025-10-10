import cv2
import numpy as np
import sys
import os

# Add path to your existing scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class SignLanguageTranslator:
    def __init__(self):
        """Initialize with YOUR actual translator"""
        print("🚀 Initializing Sign Language Translator with YOUR model...")
        
        try:
            # Import YOUR working translator
            from translator import SignLanguageTranslator as ExistingTranslator
            self.translator = ExistingTranslator()
            self.model_loaded = True
            print("✅ Successfully loaded YOUR translator!")
            
            # Test if basic methods work
            if hasattr(self.translator, 'translate'):
                print("✅ Your translator has 'translate' method")
            if hasattr(self.translator, 'process_frame'):
                print("✅ Your translator has 'process_frame' method")
                
        except Exception as e:
            print(f"❌ Error loading your translator: {e}")
            self.model_loaded = False
    
    def translate_image(self, image):
        """Translate sign language image using YOUR model"""
        try:
            if self.model_loaded:
                # Use YOUR actual translation logic
                if hasattr(self.translator, 'translate'):
                    return self.translator.translate(image)
                elif hasattr(self.translator, 'predict'):
                    return self.translator.predict(image)
                else:
                    # Try calling directly or check what methods exist
                    return "Translation in progress"
            else:
                return "Model not loaded"
                
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Error: {str(e)}"
    
    def process_frame(self, frame):
        """Process video frame using YOUR model"""
        try:
            if self.model_loaded:
                # Use YOUR frame processing logic
                if hasattr(self.translator, 'process_frame'):
                    processed_frame, translation = self.translator.process_frame(frame)
                else:
                    # Fallback: just translate without frame modification
                    translation = self.translate_image(frame)
                    processed_frame = frame
                
                # Add translation text to frame
                cv2.putText(processed_frame, translation, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                return processed_frame, translation
            else:
                # Show error on frame
                cv2.putText(frame, "Model not loaded", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame, "Model not loaded"
                
        except Exception as e:
            print(f"Frame processing error: {e}")
            cv2.putText(frame, "Processing error", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "Processing error"
    
    def get_available_signs(self):
        """Get list of signs your model supports"""
        # You might want to define this based on your model's training
        common_signs = [
            "Hello", "Thank you", "Please", "Help", 
            "Yes", "No", "Water", "Food", "Good morning",
            "How are you?", "I love you", "Sorry", "Excuse me"
        ]
        return common_signs