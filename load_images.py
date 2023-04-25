from PIL import Image
import torch
from numpy import asarray

#Class to create a dataset from the images and corresponding descriptions
class ImageDataset():
    def __init__(self, paths, transform, captions):
        self.paths = paths
        self.transform = transform
        self.captions = captions
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        self.image_path = self.paths[index]
        self.image = Image.open(self.image_path)
        cap = torch.tensor(self.captions[index], dtype=torch.float32)
        
        if self.transform:
            image_tensor = self.transform(self.image)
            
        return image_tensor, cap

