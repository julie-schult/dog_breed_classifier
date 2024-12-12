from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class CustomDataset1(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        image = Image.open(image).convert("RGB")
        image = transforms.Resize((224, 224))
        image = transforms.ToTensor()
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
        # image = transforms.Normalize(mean=[0.39113031, 0.45169107, 0.47652261], std=[0.22719704, 0.22926628, 0.23404671]) # averages of the train set

        return image, label

class CustomDataset2(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        path = self.X[idx]
        label = self.y[idx]

        image = Image.open(path).convert("RGB")
        image = transforms.Resize((224, 224))
        image = transforms.ToTensor()
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
        # image = transforms.Normalize(mean=[0.39113031, 0.45169107, 0.47652261], std=[0.22719704, 0.22926628, 0.23404671]) # averages of the train set

        return path, image, label

class CustomDataset3(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]

        image = Image.open(image).convert("RGB")
        image = transforms.Resize((224, 224))
        image = transforms.ToTensor()
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
        # image = transforms.Normalize(mean=[0.39113031, 0.45169107, 0.47652261], std=[0.22719704, 0.22926628, 0.23404671]) # averages of the train set

        return image
