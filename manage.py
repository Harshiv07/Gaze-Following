from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

@app.route('/upload')
def upload_file():
   return render_template('file_upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'
		
run_with_ngrok(app)
  
if __name__ == '__main__':
    app.run()