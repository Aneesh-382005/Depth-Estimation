import sys
import os
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))





from data.raw.KITTI.KITTI import KITTI


configPath = os.path.abspath(r"data//DatasetConfig.json")

dataset = KITTI(configPath, split="training", mode="monocular")

print(f"Total number of samples in the dataset: {len(dataset)}")


for index in range(1):
    print(dataset[index])
    sample = dataset[index]
    print("\nSample Contents:")
    print(f"Left Image Path: {os.path.join(dataset.leftImagesDirectory, dataset.leftImageFiles[index])}")




"""

image = sample["leftImage"]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow()

'''plt.imshow(image)
print(image)'''

if "labels" in sample:
    print("Labels:", sample["labels"])
else:
    print("No labels found in the dataset sample.")
"""