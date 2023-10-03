from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
from flask_cors import CORS
import pdf_struct  # <-- Newly imported
from subprocess import run


def run_pdf_struct(model_name, pdf_path):
    run(["pdf-struct", "predict", "--model", model_name, pdf_path])


def process_pdf(path_to_pdf_file, model_name):  # <-- Included here
    result = pdf_struct.predict(
        format='paragraphs',
        in_path=path_to_pdf_file,
        model=model_name
    )
    return result  # <-- Result will be whatever pdf_struct returns


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage for the parsed PDF data
parsed_pdf_data = None


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global parsed_pdf_data  # Note: Be cautious when using global variables.
    file = request.files['pdf_file']

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)

        default_model = 'PDFContractEnFeatureExtractor'
        selected_model = request.form.get('model_name', default_model)

        # After saving, proceed with parsing
        result = process_pdf(format='paragraphs', in_path=pdf_path,
                             model=selected_model)

        parsed_pdf_data = result
        return jsonify({"result": result})
    else:
        return jsonify({"error": "Invalid file format"}), 400


@app.route('/download_csv', methods=['GET'])
def download_csv():
    global parsed_pdf_data

    if parsed_pdf_data is None:
        return "No data to download"

    # Convert the parsed PDF data to a DataFrame
    df = pd.DataFrame(parsed_pdf_data)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'parsed_data.csv')
    df.to_csv(csv_path, index=False)

    return send_file(csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
