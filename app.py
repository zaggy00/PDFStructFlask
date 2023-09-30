from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pdf_struct
from flask_cors import CORS


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


CORS(app)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return 'No file part'
    file = request.files['pdf_file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        # Your PDF parsing logic here

        # Use pdf-struct to predict the structure
        result = pdf_struct.predict(
            format='paragraphs',
            in_path=pdf_path,
            model='PDFContractEnFeatureExtractor'
        )

        return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
