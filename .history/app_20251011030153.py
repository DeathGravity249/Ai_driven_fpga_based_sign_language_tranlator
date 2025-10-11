from flask import Flask, render_template, Response, jsonify
import threading
import cv2
import numpy as np

app = Flask(__name__)

# Global variables
model_loaded = False
translator = None
camera = None
is_translating = False
current_translation = ""
current_confidence = 0.0

def load_model_in_background():
    """Load your actual TensorFlow model"""
    global model_loaded, translator
    try:
        print("🔄 Loading YOUR TensorFlow model...")
        
        from ml_model.sign_detector import SignLanguageTranslator
        translator = SignLanguageTranslator()
        model_loaded = translator.model_loaded
        
        if model_loaded:
            print("✅ YOUR TENSORFLOW MODEL IS READY!")
            print("🎯 Real-time sign language detection is ACTIVE")
        else:
            print("⚠️ Model loading had issues")
            
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        model_loaded = False

def init_camera():
    global camera
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📷 Camera initialized successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Camera initialization failed: {e}")
        return False

def generate_frames():
    global camera, translator, model_loaded, is_translating, current_translation, current_confidence
    
    if camera is None:
        if not init_camera():
            placeholder = create_placeholder_image("Camera not available")
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
    
    while True:
        success, frame = camera.read()
        if not success:
            error_frame = create_placeholder_image("Camera error")
            ret, buffer = cv2.imencode('.jpg', error_frame)
        else:
            processed_frame = frame.copy()
            
            if is_translating and model_loaded and translator:
                try:
                    # CALL YOUR ACTUAL TENSORFLOW MODEL
                    gesture, confidence = translator.process_frame(frame)
                    
                    current_translation = gesture
                    current_confidence = confidence
                    
                    # Display detection results on video
                    cv2.putText(processed_frame, f"Sign: {gesture}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Confidence: {confidence:.1%}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, "LIVE DETECTION - Show Hand Signs", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                               
                except Exception as e:
                    print(f"Detection error: {e}")
                    cv2.putText(processed_frame, "Processing...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if model_loaded:
                    cv2.putText(processed_frame, "TensorFlow Model READY", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, "Loading Model...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, "Click 'Start Detection'", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def create_placeholder_image(message):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(img, message, (text_x, text_y), font, 1, (0, 0, 0), 2)
    return img

# Flask routes (same as before)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live-translate')
def live_translate():
    return render_template('live_translate.html')

@app.route('/upload-translate')
def upload_translate():
    return render_template('upload.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': model_loaded, 
        'is_translating': is_translating,
        'current_translation': current_translation,
        'current_confidence': current_confidence
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_translation')
def start_translation():
    global is_translating
    is_translating = True
    if translator:
        translator.start_detection()
    return jsonify({'status': 'success', 'message': 'Real-time TensorFlow detection started!'})

@app.route('/stop_translation')
def stop_translation():
    global is_translating
    is_translating = False
    if translator:
        translator.stop_detection()
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/get_translation')
def get_translation():
    return jsonify({
        'translation': current_translation,
        'confidence': current_confidence,
        'is_translating': is_translating
    })

if __name__ == '__main__':
    # Initialize camera
    init_camera()
    
    # Start model loading in background
    model_thread = threading.Thread(target=load_model_in_background)
    model_thread.daemon = True
    model_thread.start()
    
    print("🚀 Starting SignSpeak AI Web Interface...")
    print("📍 Open http://localhost:5000")
    print("📷 Live Translation: http://localhost:5000/live-translate")
    print("")
    print("🎯 YOUR ACTUAL TENSORFLOW MODEL IS INTEGRATED!")
    print("   - MediaPipe hand landmark detection")
    print("   - Real-time gesture classification")
    print("   - Confidence scoring")
    print("   - Professional web interface")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)