from flask import Flask
import requests
from PIL import Image
import torch
import clip
from io import BytesIO
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def inference(name):
    text = clip.tokenize([name]).to(device)
    text_features = model.encode_text(text).tolist()
    return {name:text_features}

@app.route('/<name>')
def hi(name):
    result = inference(name)
    return result

app.run(host='0.0.0.0',port=9999)
