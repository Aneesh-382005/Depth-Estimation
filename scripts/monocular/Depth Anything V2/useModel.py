import cv2
import torch
import sys
import os
import importlib.util

modulePath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Depth-Anything-V2/depth_anything_v2/dpt.py"))
moduleName = "DepthAnythingV2.depth_anything_v2.dpt"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Depth-Anything-V2/depth_anything_v2")))


spec = importlib.util.spec_from_file_location(moduleName, modulePath)
module = importlib.util.module_from_spec(spec)
sys.modules[moduleName] = module
spec.loader.exec_module(module)

DepthAnythingV2 = module.DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy