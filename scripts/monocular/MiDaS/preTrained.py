import cv2
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

modelType = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", modelType)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midasTransforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midasTransforms.dpt_transform

imagePath = r"data\\test\\friends.jpg"
image = cv2.imread(imagePath)
filename = os.path.basename(imagePath)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputBatch = transform(image).to(device)

with torch.no_grad():
    prediction = midas(inputBatch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

print(prediction.min(), prediction.max())


normalizedOutput = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX)
normalizedOutput = normalizedOutput.astype("uint8")

coloredOutput = cv2.applyColorMap(normalizedOutput, cv2.COLORMAP_VIRIDIS)

outputPath = f"data/results/MiDaS/"
os.makedirs(outputPath, exist_ok=True)
outputPath = os.path.join(outputPath, filename)

cv2.imwrite(outputPath, coloredOutput)
