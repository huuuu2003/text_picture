from pathlib import Path
import requests
from PIL import Image
import torch
import clip
from io import BytesIO
from flask import Flask
import redis
from milvus import Milvus, MetricType

milvus = Milvus(host='39.105.157.165', port='19530')
r = redis.Redis(host='39.105.157.165', port=6389,password='paSsw0o0rd')
name='picture01'
_,bool=milvus.has_collection(name)
if bool==True:
        milvus.drop_collection(name)
params = {'collection_name': name,
            'dimension': 512,
            'index_file_size': 1024,
            'metric_type': MetricType.L2}
status = milvus.create_collection(params)
print(status)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")
image_dir = Path('/root/nginx_file/tupian/lab')
base_url = 'http://39.105.157.165:9080/'
image_urls = [f"{base_url}{file.name}" for file in image_dir.glob('*.jpg')]
for i in image_urls:
    response = requests.get(i)
    if response.status_code==200:
        image=Image.open(BytesIO(response.content))
        process_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(process_image).cpu()
            vectors = image_features.tolist()
            status, ids = milvus.insert(collection_name='picture01', records=vectors)
            print(status,ids)
            value = {ids[0]: i}
            print(type(value))
            r.set(ids[0], i)
            # getValue = json.loads(r.get(ids[0]))
            print(ids)
milvus.flush(['picture01'])
milvus.close()