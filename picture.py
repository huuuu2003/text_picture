import requests
from PIL import Image
import torch
import clip
from io import BytesIO
from flask import Flask
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")

def inference(name):
    url = 'http://39.105.157.165:9080/'+name
    r = requests.get(url)
    tempIm = BytesIO(r.content)
    image = preprocess(Image.open(tempIm)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features = image_features.tolist()
    return {"picture":image_features}

@app.route('/<name>')
def hi(name):
    result = inference(name)
    return result

app.run(host='0.0.0.0',port=9999)