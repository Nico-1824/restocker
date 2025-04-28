import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
from io import BytesIO
import sys
sys.path.append('./model')
from Restocker import Restocker

app = Flask(__name__)

# Enable CORS for all routes with all options
CORS(app, origins="http://localhost:3000", supports_credentials=True, allow_headers=["Content-Type", "Authorization"])

@app.route('/identify', methods=['OPTIONS'])
def handle_options():
    response = jsonify({"message": "OK"})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
try:
    model = Restocker()
    model.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/identify', methods=['POST'])
def identify():
    try:
        data = request.json
        image_data = data.get('image')
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Convert class index to label
        classes = [
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
            'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
            'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
            'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
            'sweetpotato', 'tomato', 'turnip', 'watermelon'
        ]
        
        prediction = classes[predicted.item()]
        confidence = float(torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item())
        
        response = jsonify({
            'prediction': prediction,
            'confidence': f"{confidence:.2%}"
        })
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return response
        
    except Exception as e:
        print(f"Error in identification: {e}")
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return response, 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)