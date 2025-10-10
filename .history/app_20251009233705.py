# 


from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "🎉 Flask is working! Your web interface is ready!"

@app.route('/test')
def test():
    return "Test page is working!"

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)cl