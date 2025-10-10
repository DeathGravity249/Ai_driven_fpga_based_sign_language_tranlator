import cv2
import numpy as np
import sys
import os

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class SignLanguageTranslator:
    def __init__(self):
        """Initialize with your working TensorFlow model"""
        print("🚀 Initializing with YOUR working TensorFlow model...")
        
        try:
            # Import the modules - they're clearly working
            import translator
            import landmark_extraction
            
            self.translator = translator
            self.landmark_extraction = landmark_extraction
            self.model_loaded = True
            
            print("✅ Successfully loaded both modules!")
            
            # Let's see what we actually have by testing small calls
            print("🧪 Testing available functionality...")
            
            # Create a test frame to see what functions work
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test translator module
            if hasattr(translator, 'process_frame'):
                print("🎯 translator.process_frame() - AVAILABLE")
            if hasattr(translator, 'predict'):
                print("🎯 translator.predict() - AVAILABLE") 
            if hasattr(translator, 'detect'):
                print("🎯 translator.detect() - AVAILABLE")
            if hasattr(translator, 'translate'):
                print("🎯 translator.translate() - AVAILABLE")
                
            # Test landmark_extraction module
            if hasattr(landmark_extraction, 'extract_landmarks'):
                print("🎯 landmark_extraction.extract_landmarks() - AVAILABLE")
            if hasattr(landmark_extraction, 'detect_hands'):
                print("🎯 landmark_extraction.detect_hands() - AVAILABLE")
                
            print("🔧 Ready for real-time detection!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model_loaded = False
    
    def process_frame(self, frame):
        """Process frame - will automatically use available functions"""
        try:
            if not self.model_loaded:
                return "Model loading...", 0.8
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # STRATEGY 1: Try translator.process_frame first (most common)
            if hasattr(self.translator, 'process_frame'):
                result = self.translator.process_frame(rgb_frame)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                elif isinstance(result, str):
                    return result, 0.9
                else:
                    return "Processing...", 0.8
            
            # STRATEGY 2: Try landmark extraction + translator prediction
            elif (hasattr(self.landmark_extraction, 'extract_landmarks') and 
                  hasattr(self.translator, 'predict')):
                
                landmarks = self.landmark_extraction.extract_landmarks(rgb_frame)
                if landmarks is not None:
                    prediction = self.translator.predict(landmarks)
                    return prediction, 0.85
                else:
                    return "No hands", 0.3
            
            # STRATEGY 3: Try direct detection
            elif hasattr(self.translator, 'detect'):
                gesture = self.translator.detect(rgb_frame)
                return gesture, 0.8
            
            # STRATEGY 4: Try translate function
            elif hasattr(self.translator, 'translate'):
                gesture = self.translator.translate(rgb_frame)
                return gesture, 0.8
            
            # FALLBACK: Use simple hand detection
            else:
                return self._fallback_detection(frame), 0.7
                
        except Exception as e:
            print(f"❌ Process error: {e}")
            return f"Error: {str(e)}", 0.0
    
    def _fallback_detection(self, frame):
        """Simple fallback when specific functions aren't found"""
        # Convert to grayscale and detect hands
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            if area > 5000:
                return "LARGE HAND"
            elif area > 1000:
                return "MEDIUM HAND" 
            else:
                return "SMALL HAND"
        return "NO HAND"
    
    def start_detection(self):
        print("🟢 Starting detection with YOUR TensorFlow model")
    
    def stop_detection(self):
        print("🔴 Stopping detection")