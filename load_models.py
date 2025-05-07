import json
import torch
import torch.nn.functional as F
from model.cnn import CNN
from PIL import Image
import numpy as np


def load_model_from_config(config_path, model_dir="saved_models"):
    """Load model from configuration"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Print model info
    print(f"Loading model: {config['metadata']['type']}")
    print(f"Train accuracy: {config['metadata']['train_accuracy']:.2f}%")
    print(f"Test accuracy: {config['metadata']['test_accuracy']:.2f}%")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(device).to(device)

    # Load weights
    model_path = f"{model_dir}/{config['model_path']}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config


def preprocess_image(image_path):
    """Preprocess an image for digit recognition"""
    # Open the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to 28x28 (MNIST format)
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Invert the image if it has white digits on black background
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    # Apply MNIST normalization
    image_array = (image_array - 0.1307) / 0.3081

    # Reshape for PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(device)

    return image_tensor


def predict_digit(model, image_tensor):
    """Predict digit from image tensor"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Get confidence values
    confidences = [float(prob * 100) for prob in probabilities]

    return {
        "digit": predicted_class,
        "confidences": confidences
    }


# Example usage
if __name__ == "__main__":
    # Load models
    cnn_model, cnn_config = load_model_from_config("saved_models/cnn_model.json")
    sync_cea_model, sync_config = load_model_from_config("saved_models/sync_cea_best.json")
    async_cea_model, async_config = load_model_from_config("saved_models/async_cea_best.json")

    # Predict from an image
    image_path = "path_to_your_digit_image.png"
    image_tensor = preprocess_image(image_path)

    cnn_result = predict_digit(cnn_model, image_tensor)
    sync_result = predict_digit(sync_cea_model, image_tensor)
    async_result = predict_digit(async_cea_model, image_tensor)

    print(
        f"CNN prediction: {cnn_result['digit']} with {cnn_result['confidences'][cnn_result['digit']]:.2f}% confidence")
    print(
        f"SyncCEA prediction: {sync_result['digit']} with {sync_result['confidences'][sync_result['digit']]:.2f}% confidence")
    print(
        f"AsyncCEA prediction: {async_result['digit']} with {async_result['confidences'][async_result['digit']]:.2f}% confidence")
