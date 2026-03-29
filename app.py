import os, json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Model definition (must match training exactly) ────────────────
class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(64*8*8, 128), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH      = os.path.join('model', 'model.pth')
CLASS_PATH      = os.path.join('model', 'class_names.json')
MODEL_LOADED    = False
model           = None
class_names     = ['all', 'hem']   # fallback

if os.path.exists(MODEL_PATH):
    model = LightweightCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    MODEL_LOADED = True
    print('Model loaded from', MODEL_PATH)
else:
    print('WARNING: model.pth not found — predictions will return demo values.')

if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH) as f:
        class_names = json.load(f)

# ── Image transform ───────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    if not MODEL_LOADED:
        # Demo mode — return plausible fake values
        import random
        p = round(random.uniform(0.55, 0.95), 4)
        label_idx = 0 if p > 0.5 else 1
        return jsonify({
            'demo_mode': True,
            'prediction': class_names[label_idx].upper(),
            'label_index': label_idx,
            'probabilities': {
                class_names[0]: round(p, 4),
                class_names[1]: round(1 - p, 4)
            },
            'confidence': round(max(p, 1-p) * 100, 1)
        })

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    prob_dict = {class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))}
    print(f"Raw output: {output}")
    print(f"Probs: {prob_dict}")
    print(f"Predicted index: {pred_idx} → {class_names[pred_idx]}")

    return jsonify({
        'demo_mode': False,
        'prediction': class_names[pred_idx].upper(),
        'label_index': pred_idx,
        'probabilities': prob_dict,
        'confidence': round(probs[pred_idx].item() * 100, 1)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL_LOADED, 'classes': class_names})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
