
from data.raw.KITTI.KITTI import KITTIDataset
configPath = "data\DatasetConfig.json"

# Initialize the dataset
dataset = KITTIDataset(configPath, split="training", mode="monocular")

# Fetch a sample
sample = dataset[0]
print("Left Image:", sample["left_image"])
print("Labels:", sample.get("labels"))
