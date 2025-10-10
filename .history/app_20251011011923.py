from flask import Flask, render_template, Response
import threading
import cv2

app = Flask(__name__)

# Global variables
model_loaded = False
camera = None

def load_model_in_background():
    """Load the model in background without blocking Flask"""
    global model_loaded
    try:
        print("🔄 Loading AI model in background...")
        
        # Add your model loading code here
        import sys
        sys.path.append('scripts')
        
        from ml_model.sign_detector import SignLanguageTranslator
        translator = SignLanguageTranslator()
        model_loaded = True
        print("✅ AI model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️  Model loading: {e}")

def init_camera():
    """Initialize camera"""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📷 Camera initialized successfully!")
    except Exception as e:
        print(f"⚠️  Camera initialization failed: {e}")

def generate_frames():
    """Generate video frames for streaming"""
    global camera
    if camera is None:
        init_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # You can add your sign language processing here later
            # For now, just stream the raw camera feed
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
    return {'model_loaded': model_loaded, 'status': 'Web interface ready!'}

# Add the video feed endpoint
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_translation')
def start_translation():
    """Endpoint to start sign language translation"""
    # Add your translation logic here
    return {'status': 'translation_started', 'message': 'Translation started'}

@app.route('/stop_translation')
def stop_translation():
    """Endpoint to stop sign language translation"""
    # Add your stop logic here
    return {'status': 'translation_stopped', 'message': 'Translation stopped'}

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
    print("🖼️  Upload Images: http://localhost:5000/upload-translate")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)