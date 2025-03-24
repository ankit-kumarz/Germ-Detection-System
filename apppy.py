import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def detect_germs(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    germ_area = sum(cv2.contourArea(c) for c in contours)
    return germ_area, contours

def visualize_germs(image, contours):
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output

def calculate_cleanliness(before_area, after_area):
    if before_area == 0:
        return 100.0  # Already clean
    removal_ratio = (before_area - after_area) / before_area
    return max(0, min(removal_ratio * 100, 100))

def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/process', methods=['POST'])
def process_images():
    
    before_file = request.files['before']
    after_file = request.files['after']

    before_image = cv2.imdecode(np.frombuffer(before_file.read(), np.uint8), cv2.IMREAD_COLOR)
    after_image = cv2.imdecode(np.frombuffer(after_file.read(), np.uint8), cv2.IMREAD_COLOR)

    
    preprocessed_before = preprocess_image(before_image)
    before_area, before_contours = detect_germs(preprocessed_before)

    preprocessed_after = preprocess_image(after_image)
    after_area, after_contours = detect_germs(preprocessed_after)

    
    before_visualization = visualize_germs(before_image, before_contours)
    after_visualization = visualize_germs(after_image, after_contours)

    
    cleanliness_score = calculate_cleanliness(before_area, after_area)

   
    response_data = {
        "cleanliness_score": cleanliness_score,
        "before_visualization": encode_image(before_visualization),
        "after_visualization": encode_image(after_visualization)
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
