from flask import Flask, render_template, request, jsonify, Response
import cv2
import base64
import numpy as np
import os
import sys

# Add scripts directory to path to import your existing modules
sys.path.append('scripts')

from ml_model.sign_detector import SignLanguageTranslator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize your existing translator
translator = SignLanguageTranslator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live-translate')
def live_translate():
    return render_template('live_translate.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    image_file = request.files['image']
    
    # Convert image to OpenCV format
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Use your existing translation logic
    try:
        # This will call your existing scripts/translator.py
        translation = translator.translate_image(image)
        
        return jsonify({
            'status': 'success',
            'translation': translation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)