import os
import base64
from flask import Flask, request, jsonify
import cv2
import pywt
import numpy as np
import json
import joblib
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load model and class dictionary
model = None
class_dict = {}

def load_data():
    global model, class_dict
    with open("./class_dictonary.json", "r") as f:
        number = json.load(f)
        class_dict = {v: k for k, v in number.items()}
    with open("./model.pkl", "rb") as f:
        model = joblib.load(f)
    print("Completed loading data...")

def w2d(img, mode='haar', level=1):
    """Apply wavelet transform to extract high-frequency components."""
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Remove low-frequency components
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

def from_b64_to_cv2_image(data):
    """Convert base64-encoded image to OpenCV format."""
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_faces(img):
    """Detect and crop faces from an image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:  # Accept faces with at least 2 detected eyes
            cropped_faces.append(roi_color)
    
    return cropped_faces

def classify_image(image_b64_data):
    """Classify an image based on pre-trained model."""
    results = []
    img = from_b64_to_cv2_image(image_b64_data)
    faces = get_cropped_faces(img)

    for face in faces:
        if face is None:
            continue
        scaled_raw_img = cv2.resize(face, (32, 32))
        img_har = w2d(face, "db1", 5)
        scaled_har_img = cv2.resize(img_har, (32, 32))
        scaled_har_img = np.stack([scaled_har_img] * 3, axis=-1)

        combined_image = np.vstack((
            scaled_raw_img.reshape(32 * 32 * 3, 1),
            scaled_har_img.reshape(32 * 32 * 3, 1)
        ))
        final_img = np.array(combined_image).reshape(1, -1).astype(float)

        prediction = model.predict(final_img)[0]
        results.append({'class':class_dict[prediction],
                        'class_prob':(model.predict_proba(final_img)*100).tolist()[0]})

    return results

@app.route('/classify_image', methods=['POST'])
def classify_image_api():
    """API Endpoint: Accepts base64 image and returns the predicted class."""
    try:
        print("called");
        data=request.get_json();
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        result = classify_image(image_data)
        if not result:
            return jsonify({"error": "No face detected"}), 400

        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

if __name__ == "__main__":
    load_data()
    # print("Checking Haarcascade:", os.path.exists(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

    # # Convert test image to base64 before passing it
    # test_image_path = os.path.join("test_images", "virat3.jpg")
    # if os.path.exists(test_image_path):
    #     image_b64 = image_to_base64(test_image_path)
    #     print(classify_image(image_b64))
    # else:
    #     print(f"Error: Test image '{test_image_path}' not found.")

    app.run(port=5000, debug=True)
