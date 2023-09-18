import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.split = 'train' if train else 'test'
        self.data = self.load_data()

    def load_data(self):
        data = []
        a_dir_image = os.path.join(self.root_dir, 'CAR-A', f'a_{self.split}_images')
        a_gt = os.path.join(self.root_dir, 'CAR-A', f'a_{self.split}_gt.txt')
        b_dir_image = os.path.join(self.root_dir, 'CAR-B', f'b_{self.split}_images')
        b_gt = os.path.join(self.root_dir, 'CAR-B', f'b_{self.split}_gt.txt')
        for (gt_file, image_dir) in [(a_gt, a_dir_image), (b_gt, b_dir_image)]:
            with open(gt_file, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                image_name, label = line.split()
                image_path = os.path.join(image_dir, image_name)
                label_length = len(label)
                target = torch.full(size=(8,), fill_value=10, dtype=torch.long)
                target[:label_length] = torch.tensor([int(digit) for digit in label])
                label_length = torch.tensor(label_length, dtype=torch.long)
                data.append((image_path, label_length, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, length, target = self.data[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, length, target
