from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import re
import json
import os

# Import the required model classes
from model.cnn import CNN
from automata.fsm import SyncCEA, AsyncCEA

# Create Flask app with the correct static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saved models if it doesn't exist
os.makedirs("saved_models", exist_ok=True)


# Load pre-trained models
def load_models():
    models = {}
    model_metrics = {}

    # Try to load metrics
    try:
        metrics_path = os.path.join('static', 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
                print("Loaded model metrics from file")
    except Exception as e:
        print(f"Warning: Could not load metrics: {e}")
        model_metrics = {}

    try:
        # Load standard CNN model
        cnn_model = CNN(device).to(device)
        cnn_model.load_state_dict(torch.load('saved_models/cnn_model.pth', map_location=device))
        models['cnn'] = cnn_model
        print("Loaded CNN model")

        # Load best SyncCEA model
        sync_cea_model = CNN(device).to(device)
        sync_cea_model.load_state_dict(torch.load('saved_models/sync_cea_best.pth', map_location=device))
        models['syncCEA'] = sync_cea_model
        print("Loaded SyncCEA model")

        # Load best AsyncCEA model
        async_cea_model = CNN(device).to(device)
        async_cea_model.load_state_dict(torch.load('saved_models/async_cea_best.pth', map_location=device))
        models['asyncCEA'] = async_cea_model
        print("Loaded AsyncCEA model")

    except Exception as e:
        print(f"Warning: {e}")
        print("Using random models for demonstration.")

        # For demonstration, create random models if saved ones are not found
        models['cnn'] = CNN.create_random_model(device)
        models['syncCEA'] = CNN.create_random_model(device)
        models['asyncCEA'] = CNN.create_random_model(device)

    return models, model_metrics


# Initialize models
models, model_metrics = load_models()


# Preprocess the image
def preprocess_image(image_data):
    try:
        # Extract the base64 encoded data
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

        # Resize to 28x28 (MNIST format)
        image = image.resize((28, 28))

        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Invert the image (MNIST has white digits on black background)
        image_array = 1.0 - image_array

        # Apply MNIST normalization
        image_array = (image_array - 0.1307) / 0.3081

        # Reshape for PyTorch model
        image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(device)

        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


# Route to serve the main index.html page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# API endpoint for recognizing digits
@app.route('/api/recognize', methods=['POST'])
def recognize_digit():
    if not request.json or 'image' not in request.json or 'modelType' not in request.json:
        return jsonify({'error': 'Missing image data or model type'}), 400

    image_data = request.json['image']
    model_type = request.json['modelType']

    if model_type not in models:
        return jsonify({'error': f'Invalid model type: {model_type}'}), 400

    try:
        # Preprocess the image
        image_tensor = preprocess_image(image_data)

        # Make prediction
        model = models[model_type]
        model.eval()

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class = int(torch.argmax(outputs, dim=1).item())

        # Convert confidence values to percentage
        confidences = [float(prob * 100) for prob in probabilities]

        return jsonify({
            'digit': predicted_class,
            'confidences': confidences
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'models': list(models.keys()),
        'device': str(device)
    })


@app.route('/api/model_metrics', methods=['GET'])
def get_model_metrics():
    return jsonify(model_metrics)

# Run the Flask app
if __name__ == '__main__':
    print(f"Starting Flask server using {device} device")
    print(f"Access the application at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
