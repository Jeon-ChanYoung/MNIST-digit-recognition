from flask import Flask, request, render_template, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from CNN_model import CNN1

app = Flask(__name__)

# 모델 정의 및 로드
model = CNN1()
model.load_state_dict(torch.load('model_parameter/CNN_Model2_MNIST.pt'))
model.eval()

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    image = preprocess_image(image)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)

    return jsonify({'digit': prediction.item()})

if __name__ == '__main__':
    app.run(debug=True)
