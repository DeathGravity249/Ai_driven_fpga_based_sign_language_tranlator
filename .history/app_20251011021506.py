# from flask import Flask, render_template, Response, jsonify
# import threading
# import cv2
# import time
# import numpy as np

# app = Flask(__name__)

# # Global variables
# model_loaded = False
# translator = None
# camera = None
# is_translating = False
# current_translation = ""

# def load_model_in_background():
#     """Load the model in background without blocking Flask"""
#     global model_loaded, translator
#     try:
#         print("🔄 Loading AI model in background...")
        
#         # Import your working script
#         import sys
#         import os
#         sys.path.append('scripts')
        
#         # Import your actual working model
#         try:
#             from ml_model.sign_detector import SignLanguageTranslator
#             translator = SignLanguageTranslator()
#             print("✅ AI model loaded successfully!")
#             model_loaded = True
            
#         except ImportError as e:
#             print(f"⚠️  Could not import model: {e}")
#             # If import fails, create a mock for testing
#             class MockTranslator:
#                 def process_frame(self, frame):
#                     # This is where your actual detection happens
#                     # For now, return mock data
#                     return "Hello", 0.95
#                 def start_detection(self):
#                     print("🟢 Detection started")
#                 def stop_detection(self):
#                     print("🔴 Detection stopped")
                    
#             translator = MockTranslator()
#             model_loaded = True
#             print("🔧 Using mock translator for demonstration")
            
#     except Exception as e:
#         print(f"❌ Model loading failed: {e}")
#         model_loaded = False

# def init_camera():
#     """Initialize camera"""
#     global camera
#     try:
#         camera = cv2.VideoCapture(0)
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         print("📷 Camera initialized successfully!")
#         return True
#     except Exception as e:
#         print(f"⚠️  Camera initialization failed: {e}")
#         return False

# def generate_frames():
#     """Generate video frames with gesture detection"""
#     global camera, translator, model_loaded, is_translating, current_translation
    
#     if camera is None:
#         if not init_camera():
#             placeholder = create_placeholder_image("Camera not available")
#             ret, buffer = cv2.imencode('.jpg', placeholder)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             return
    
#     while True:
#         success, frame = camera.read()
#         if not success:
#             error_frame = create_placeholder_image("Camera error")
#             ret, buffer = cv2.imencode('.jpg', error_frame)
#         else:
#             # Process frame for gesture detection if translation is active
#             processed_frame = frame.copy()
            
#             if is_translating and model_loaded and translator:
#                 try:
#                     # Call your actual gesture detection function here
#                     gesture, confidence = translator.process_frame(frame)
                    
#                     if gesture and confidence > 0.7:  # Adjust confidence threshold as needed
#                         current_translation = f"{gesture} ({confidence:.1%})"
                        
#                         # Add text overlay to the frame
#                         cv2.putText(processed_frame, f"Sign: {gesture}", (10, 30), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                         cv2.putText(processed_frame, f"Confidence: {confidence:.1%}", (10, 70), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
#                     # Add translation status overlay
#                     status_text = "TRANSLATING - Show signs to camera"
#                     cv2.putText(processed_frame, status_text, (10, 110), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                               
#                 except Exception as e:
#                     print(f"Gesture detection error: {e}")
#                     cv2.putText(processed_frame, "Detection Error", (10, 30), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             else:
#                 # Show instruction when not translating
#                 cv2.putText(processed_frame, "Click 'Start Translation' to begin", (10, 30), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             ret, buffer = cv2.imencode('.jpg', processed_frame)
        
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# def create_placeholder_image(message):
#     """Create a placeholder image when camera is not available"""
#     img = np.zeros((480, 640, 3), dtype=np.uint8)
#     img.fill(200)  # Light gray background
    
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_size = cv2.getTextSize(message, font, 1, 2)[0]
#     text_x = (640 - text_size[0]) // 2
#     text_y = (480 + text_size[1]) // 2
    
#     cv2.putText(img, message, (text_x, text_y), font, 1, (0, 0, 0), 2)
#     return img

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/live-translate')
# def live_translate():
#     return render_template('live_translate.html')

# @app.route('/upload-translate')
# def upload_translate():
#     return render_template('upload.html')

# @app.route('/api/status')
# def status():
#     return jsonify({
#         'model_loaded': model_loaded, 
#         'status': 'Web interface ready!',
#         'camera_available': camera is not None and camera.isOpened(),
#         'is_translating': is_translating,
#         'current_translation': current_translation
#     })

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), 
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/start_translation')
# def start_translation():
#     """Endpoint to start sign language translation"""
#     global is_translating, current_translation
    
#     if not model_loaded:
#         return jsonify({
#             'status': 'error', 
#             'message': 'AI model is still loading. Please wait...'
#         })
    
#     is_translating = True
#     current_translation = "Detection started - show signs to camera"
    
#     # Call your model's start method if it exists
#     if hasattr(translator, 'start_detection'):
#         translator.start_detection()
    
#     return jsonify({
#         'status': 'success', 
#         'message': 'Real-time gesture detection started!',
#         'model_ready': model_loaded
#     })

# @app.route('/stop_translation')
# def stop_translation():
#     """Endpoint to stop sign language translation"""
#     global is_translating, current_translation
    
#     is_translating = False
#     current_translation = "Translation stopped"
    
#     # Call your model's stop method if it exists
#     if hasattr(translator, 'stop_detection'):
#         translator.stop_detection()
    
#     return jsonify({
#         'status': 'success', 
#         'message': 'Translation stopped'
#     })

# @app.route('/get_translation')
# def get_translation():
#     """Get current translation result"""
#     return jsonify({
#         'translation': current_translation,
#         'is_translating': is_translating
#     })

# if __name__ == '__main__':
#     # Initialize camera
#     init_camera()
    
#     # Start model loading in background
#     model_thread = threading.Thread(target=load_model_in_background)
#     model_thread.daemon = True
#     model_thread.start()
    
#     print("🚀 Starting SignSpeak AI Web Interface...")
#     print("📍 Open http://localhost:5000")
#     print("📷 Live Translation: http://localhost:5000/live-translate")
#     print("🖼️  Upload Images: http://localhost:5000/upload-translate")
#     print("🎯 Real-time gesture detection will work when you click 'Start Translation'")
    
#     app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

