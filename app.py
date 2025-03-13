from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from predict import predict_hindi
from english_predict import predict_audio_class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Check if folder exists at startup
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about_me.html')


@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/process_scan', methods=['POST'])
def process_scan():
    audio_file = request.files.get('audio_file')
    emotion = request.form.get('emotion')
    speech = request.form.get('speech')
    print(emotion,speech)
    print(audio_file)
    if not audio_file:
        return "No file uploaded", 400

    if allowed_file(audio_file.filename):
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        if speech == 'Hindi':
            result = predict_hindi(filepath)
        else:
            result = predict_audio_class(filepath)
            result = result[0]
        print(result)
        # Process the file and emotion as needed
        return render_template('scan.html',prediction=result,emotion=emotion)
    else:
        return "File type not allowed", 40

if __name__ == '__main__':
    app.run(debug=True)