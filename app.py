import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('ERROR')

# Load the pre-trained model
model = tf.keras.models.load_model('simple_keras_model.h5', compile=False)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if input data is present
        if 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400

        input_data = data['input']

        # Convert input data to numpy array
        input_array = np.array(input_data)

        # Make prediction
        predictions = model.predict(input_array)

        # Convert predictions to list of floats
        results = predictions.tolist()

        # Return the prediction as JSON
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
