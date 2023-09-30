from flask import Flask, request, jsonify
import pdf_struct

app = Flask(__name__)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    uploaded_file = request.files['pdf_file']
    if uploaded_file.filename != '':
        pdf_path = f"./uploaded_files/{uploaded_file.filename}"
        uploaded_file.save(pdf_path)

        # Use pdf-struct to predict the structure
        result = pdf_struct.predict(
            format='paragraphs',
            in_path=pdf_path,
            model='PDFContractEnFeatureExtractor'
        )

        return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
