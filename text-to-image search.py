import urllib.request
from io import BytesIO
import cv2
import requests
import numpy as np
from PIL import Image
from milvus import Milvus
import redis
name = input('查询的文字:')
url = 'http://39.105.157.165:9999/'+f'{name}'
r = requests.get(url)
r_json = r.json()
text_c = r_json.get(f'{name}')
milvus = Milvus(host='39.105.157.165', port='19530')
r = redis.Redis(host='39.105.157.165', port=6389, decode_responses=True,password='paSsw0o0rd')
name = 'picture01'
status,ips = milvus.search(collection_name=name,query_records=text_c,top_k=3)
img_name = r.get(ips.id_array[0][0])
# res = urllib.request.urlopen(img_name)
# img_f = np.asarray(bytearray(res.read()), dtype=np.uint8)
# img = cv2.imdecode(img_f, cv2.IMREAD_COLOR)
# cv2.imshow(f'{name}',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

r = requests.get(img_name)
tempIm = BytesIO(r.content)
img = Image.open(tempIm)
img.show()