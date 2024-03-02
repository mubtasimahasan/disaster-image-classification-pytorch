import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

from utils.config import CFG

class ImageFolderCustom(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        self.load_data()

    def load_data(self):
        class_names = os.listdir(self.root)  
        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    continue
                self.data.append(image)
                self.targets.append(float(image_name[1:2])-1)

    def __getitem__(self, index: int) -> tuple:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)


def get_data(path: str) -> Tuple[torch.utils.data.Dataset, List[str]]:
    transform = transforms.Compose([
        transforms.Resize(CFG.input_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3] if x.size(0) > 3 else x),
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.size(0) == 1 else x)
    ])
    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
    
    dataset = ImageFolderCustom(path, transform=transform, target_transform=target_transform)
    
    return dataset


def get_dataloader(dataset: torch.utils.data.Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:    
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, CFG.split)
    
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    
    return train_dataloader, test_dataloader, valid_dataloader


def display_random_images(dataset: torch.utils.data.dataset.Dataset):
    n = 5
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(15, 15))
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample]  
        targ_image_adjust = targ_image.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        title = f"class: {CFG.classes[(targ_label)]}"
        plt.title(title)

