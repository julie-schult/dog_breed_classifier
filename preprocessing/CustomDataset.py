from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        imgage = self.X[idx]
        label = self.y[idx]

        image = Image.open(image).convert("RGB")
        image = transforms.Resize((225, 225))
        image = transforms.ToTensor()
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
        # image = transforms.Normalize(mean=[0.39113031, 0.45169107, 0.47652261], std=[0.22719704, 0.22926628, 0.23404671]) # averages of the train set

        return image, label
