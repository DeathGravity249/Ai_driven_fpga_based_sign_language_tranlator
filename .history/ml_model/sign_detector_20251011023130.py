import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import sys
import os

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class SignLanguageTranslator:
    def __init__(self):
        """Initialize with your actual TensorFlow model"""
        print("🚀 Initializing with YOUR actual TensorFlow model...")
        
        try:
            # Load model & labels (same as your working script)
            self.model = tf.keras.models.load_model("saved_models/landmark_model.h5")
            with open("saved_models/landmark_labels.json", "r") as f:
                self.class_names = json.load(f)
            
            # Initialize MediaPipe (same as your working script)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, 
                max_num_hands=1, 
                min_detection_confidence=0.7
            )
            
            self.model_loaded = True
            print(f"✅ TensorFlow model loaded successfully!")
            print(f"✅ Class names: {self.class_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def process_frame(self, frame):
        """Process frame using YOUR actual detection logic"""
        try:
            if not self.model_loaded:
                return "Model not loaded", 0.0
            
            # Flip frame for mirror view (same as your script)
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (same as your script)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Get the first hand landmarks
                lm = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks (same as your script)
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Extract coordinates and predict (same as your script)
                coords = []
                for lm_point in lm.landmark:
                    coords.extend([lm_point.x, lm_point.y, lm_point.z])
                coords = np.array(coords).reshape(1, -1)  # shape (1, 63)
                
                # Make prediction (same as your script)
                pred = self.model.predict(coords, verbose=0)[0]
                idx = np.argmax(pred)
                label = self.class_names[idx]
                confidence = pred[idx]
                
                # Return the detected gesture and confidence
                return label, float(confidence)
                
            else:
                # No hand detected
                return "Show your hand", 0.0
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return "Detection error", 0.0
    
    def start_detection(self):
        print("🟢 Starting real-time TensorFlow detection")
    
    def stop_detection(self):
        print("🔴 Stopping detection")