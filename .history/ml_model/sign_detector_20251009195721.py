import cv2
import numpy as np
import sys
import os

# Add path to your existing scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Import your existing modules
try:
    from translator import SignLanguageTranslator as ExistingTranslator
    from landmark_extraction import extract_landmarks
    print("Successfully imported existing modules")
except ImportError as e:
    print(f"Import warning: {e}")

class SignLanguageTranslator:
    def __init__(self):
        """Initialize using your existing translator"""
        try:
            # Initialize your existing translator
            self.translator = ExistingTranslator()
            self.model_loaded = True
        except:
            print("Using mock translator - integrate with your actual model")
            self.model_loaded = False
    
    def translate_image(self, image):
        """Translate sign language image using your existing code"""
        if self.model_loaded:
            # Use your existing translation logic
            return self.translator.translate(image)
        else:
            # Mock translation for testing
            return self._mock_translation()
    
    def process_frame(self, frame):
        """Process video frame for real-time translation"""
        try:
            # Your existing frame processing logic
            if self.model_loaded:
                landmarks = extract_landmarks(frame)
                translation = self.translator.translate_from_landmarks(landmarks)
            else:
                translation = self._mock_translation()
            
            # Add translation to frame for display
            cv2.putText(frame, translation, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return frame, translation
            
        except Exception as e:
            print(f"Translation error: {e}")
            return frame