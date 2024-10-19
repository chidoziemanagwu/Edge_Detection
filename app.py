from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image is uploaded
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']

        if file.filename == '':
            return 'No file selected', 400

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Perform edge detection
            process_image(file.filename)

            return redirect(url_for('display_result', filename=file.filename))

    return render_template('upload.html')

def process_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply edge detection (Sobel and Canny)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(image=image, threshold1=100, threshold2=200)

    # Save outputs
    sobelx_path = os.path.join(app.config['OUTPUT_FOLDER'], 'sobelx_' + filename)
    sobely_path = os.path.join(app.config['OUTPUT_FOLDER'], 'sobely_' + filename)
    edges_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edges_' + filename)

    cv2.imwrite(sobelx_path, sobelx)
    cv2.imwrite(sobely_path, sobely)
    cv2.imwrite(edges_path, edges)

@app.route('/result/<filename>')
def display_result(filename):
    sobelx_image = 'outputs/sobelx_' + filename
    sobely_image = 'outputs/sobely_' + filename
    edges_image = 'outputs/edges_' + filename

    return render_template('result.html', original_image='uploads/' + filename,
                           sobelx_image=sobelx_image,
                           sobely_image=sobely_image,
                           edges_image=edges_image)

if __name__ == '__main__':
    app.run(debug=True)
