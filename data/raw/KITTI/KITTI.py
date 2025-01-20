import os
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class KITTI(Dataset):
    def __init__(self, configPath, split = "training", mode = "monocular", transforms = None):

        """
        Arguments: 
            configPath (str): Path to the JSON configuration file.
            split (str): Dataset split to use ('train' or 'test').
            mode (str): Mode of operation ('monocular' or 'stereo').
            transforms (callable, optional): Transformations to apply to images.
        """

        with open(configPath) as file:
            config = json.load(file)

        self.datasetName = "KITTI" 
        KITTIconfig = config["KITTI"]   
        self.root = KITTIconfig["root"]
        self.split = split
        self.mode = mode
        self.transforms = transforms

        splitConfig = KITTIconfig["splits"][split]
        self.leftImagesDirectory = os.path.join(self.root, splitConfig["leftImages"])
        self.rightImagesDirectory = os.path.join(self.root, splitConfig["rightImages"])
        self.calibrationDirectory = os.path.join(self.root, splitConfig["calibration"])
        self.labelsDirectory = os.path.join(self.root, splitConfig["labels"])

        self.leftImageFiles = sorted(os.listdir(self.leftImagesDirectory))
        self.rightImageFiles = sorted(os.listdir(self.rightImagesDirectory)) if self.rightImagesDirectory else None
        self.calibrationFiles = sorted(os.listdir(self.calibrationDirectory)) if self.calibrationDirectory else None
        self.labelFiles = sorted(os.listdir(self.labelsDirectory)) if self.labelsDirectory else None

    def __len__(self):
        return len(self.leftImageFiles)

    def __getitem__(self, index):

        """
        Arguments:
            index (int): Index of the image to retrieve.
        Returns:
            dict: A dictionary containing image(s), labels, and calibration data.
        """

        if index >= len(self):
            raise IndexError("Index out of range.")

        leftImagesPath = os.path.join(self.leftImagesDirectory, self.leftImageFiles[index])
        leftImage = Image.open(leftImagesPath).convert("RGB")
        data = {"leftImage": leftImage}

        if self.mode == "stereo":
            rightImagesPath = os.path.join(self.rightImagesDirectory, self.rightImageFiles[index])
            rightImage = Image.open(rightImagesPath).convert("RGB")
            data["rightImage"] = rightImage

        if self.calibrationDirectory:
            calibrationPath = os.path.join(self.calibrationDirectory, self.calibrationFiles[index])
            calibration = self.loadCalibration(calibrationPath)
            data["calibration"] = calibration

        if self.labelsDirectory:
            labelPath = os.path.join(self.labelsDirectory, self.labelFiles[index])
            labels = self.loadLabels(labelPath)
            data["labels"] = labels

        if self.transforms:
            data["leftImage"] = self.transforms(data["leftImage"])

            if "rightImage" in data:
                data["rightImage"] = self.transforms(data["rightImage"])

        return data

    def loadCalibration(self, calibrationPath):
        
        """
        Arguments:
            calibrationPath (str): Path to the calibration file.

        Returns:
            dict: Calibration matrices.
        """

        calibration = {}
        with open(calibrationPath) as file:
            for line in file:
                if ":" not  in line:
                    continue
                key, value = line.split(":", 1)
                calibration[key] = np.array([float(x) for x in value.split()])
        return calibration
    
    def loadLabels(self, LabelPath):

        """
        Arguments:
            labelPath (str): Path to the label file.

        Returns:
            dict: List of objects with their label attributes.
        """

        labels = []

        with open(LabelPath) as file:
            for line in file:
                parts = line.split()

                label = {
                    "type": parts[0],
                    "truncated": float(parts[1]),
                    "occluded": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bbox": [float(x) for x in parts[4:8]],
                    "dimensions": [float(x) for x in parts[8:11]],
                    "location": [float(x) for x in parts[11:14]],
                    "rotation_y": float(parts[14])
                }

                labels.append(label)

        return labels
