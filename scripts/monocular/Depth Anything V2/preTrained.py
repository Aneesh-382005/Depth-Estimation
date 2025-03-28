from transformers import pipeline
from PIL import Image
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
image = Image.open(r'data\\test\\friends.jpg')
depth = pipe(image)["depth"]
path = r'data\\results\\DepthAnythingV2\\friends.jpg'
os.makedirs(os.path.dirname(path), exist_ok=True)
depth.save(path)