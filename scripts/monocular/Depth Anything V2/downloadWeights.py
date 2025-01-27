import os
import urllib.request
from tqdm import tqdm

url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
checkpoints = "checkpoints"
outputPath = os.path.join(checkpoints, "depth_anything_v2_vitl.pth")

os.makedirs(checkpoints, exist_ok=True)

class DownloadProgressBar(tqdm):
    def updateTo(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

if not os.path.exists(outputPath):
    print("Downloading pre-trained weights...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="depth_anything_v2_vitl.pth") as t:
        urllib.request.urlretrieve(url, outputPath, reporthook=t.updateTo)
    print(f"Download complete! File saved to {outputPath}")
else:
    print(f"File already exists at {outputPath}. Skipping download.")
