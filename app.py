import os
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set the device: use CUDA if available otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing pipeline
val_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    """Load the pre-trained EfficientNet model"""
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_file):
    """Preprocess uploaded image for model inference"""
    try:
        image = Image.open(image_file).convert("RGB")
        image_tensor = val_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_image(model, image_tensor):
    """Perform inference and return HCQ toxicity probability"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        hcq_toxicity_probability = probabilities[:, 1].item()
    return hcq_toxicity_probability

# Load model on startup
print("Loading model...")
try:
    model = load_model('hcquery_model.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            print(f"Processing file: {file.filename}")
            
            # Preprocess the image
            image_tensor = preprocess_image(file)
            if image_tensor is None:
                print("Image preprocessing failed")
                return jsonify({'error': 'Failed to process image'}), 400
            
            print("Image preprocessed successfully, making prediction...")
            
            # Make prediction
            probability = predict_image(model, image_tensor)
            print(f"Prediction made: {probability}")
            
            # Convert probability to percentage and create risk assessment
            percentage = probability * 100
            
            if percentage < 50:
                risk_level = "Low"
                risk_color = "#28a745"  # Green
                message = "Low likelihood of HCQ retinopathy"
            else:
                risk_level = "High"
                risk_color = "#dc3545"  # Red
                message = "High likelihood of HCQ retinopathy"
            
            return jsonify({
                'probability': round(percentage, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'message': message
            })
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
