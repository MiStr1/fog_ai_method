import requests
from shutil import make_archive
from os import path, remove

make_archive('test_train', 'zip', 'test_train')

with open("test_train.zip", "rb") as file:
    url = 'http://127.0.0.1:5000/trainAI'
    r = requests.post(url, data=file, headers={'Content-Type': 'application/octet-stream'})

if path.exists("test_train.zip"):
    remove("test_train.zip")