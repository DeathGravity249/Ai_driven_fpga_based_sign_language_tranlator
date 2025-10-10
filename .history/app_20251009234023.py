from flask import Flask, render_template
import threading

app = Flask(__name__)

# Global variable to track model loading
model_loaded = False

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

if __name__ == '__main__':
    # Start model loading in background
    model_thread = threading.Thread(target=load_model_in_background)
    model_thread.daemon = True
    model_thread.start()
    
    print("🚀 Starting SignSpeak AI Web Interface...")
    print("📍 Open http://localhost:5000")
    print("📷 Live Translation: http://localhost:5000/live-translate")
    print("🖼️  Upload Images: http://localhost:5000/upload-translate")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)