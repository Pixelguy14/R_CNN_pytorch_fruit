import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, root, ann_dir, transform=None, classes=None):
        self.root = root
        self.ann_dir = ann_dir
        self.transform = transform
        self.imgs = list(sorted(os.listdir(root)))
        self.annotations = list(sorted(os.listdir(ann_dir)))
        self.class_map = {name: i for i, name in enumerate(classes)}

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        ann_path = os.path.join(self.ann_dir, self.annotations[index])
        
        img = Image.open(img_path).convert("RGB")
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            label_idx = self.class_map[label]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])
