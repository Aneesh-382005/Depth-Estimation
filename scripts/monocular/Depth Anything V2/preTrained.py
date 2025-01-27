import torch
import torchvision.transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel
model = AutoModel.from_pretrained("depth-anything/Depth-Anything-V2")
model.eval()
