from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure a file is provided in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Retrieve the image file
        file = request.files['file']
        img = image.load_img(file, target_size=(64, 64))  # Match the input size of the model

        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)
        class_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Return the result
        return jsonify({'prediction': class_label, 'confidence': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
