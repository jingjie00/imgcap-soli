import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os

class SiameseDataset(Dataset):
    def __init__(self, base_dir, feature_extractor):
        """
        Args:
            base_dir (string): Directory with all the image subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.base_dir = base_dir
        self.resolutions = ['flickr8k_images', 'R0.1S1', 'R0.1S50', 'R0.2S1', 'R0.2S50', 'R0.5S1', 'R0.5S50', 'R0.05S50', 'R0.5S1_GF500','R1S1_GF500']
        self.feature_extractor = feature_extractor
        
        with open(base_dir+'Flickr_8k.trainImages.txt','r') as f:
            self.image_names = f.read().splitlines()
        
    def __len__(self):
        # Assume all resolutions have the same number of images
        return len(self.image_names)
    
    def resize_if_needed(image, min_dimension=1):
        """Resizes an image if its width or height is less than the minimum dimension.

        Args:
            image: The PIL image object to resize.
            min_dimension: The minimum required dimension (default: 1).

        Returns:
            The resized image object if resized, otherwise the original image.
        """
        width, height = image.size
        if width < min_dimension or height < min_dimension:
            # Resize the image to maintain aspect ratio
            new_size = (max(min_dimension, width), max(min_dimension, height))
            image = image.resize(new_size, Image.ANTIALIAS)
        return image
    
    def __getitem__(self, idx):

        # anchor
        res1 = random.choice(self.resolutions)
        anchor_path = os.path.join(self.base_dir,res1, self.image_names[idx])
        
        # Select an dimension apart from the anchor
        resolutions_temp = self.resolutions.copy()
        resolutions_temp.remove(res1)
        res2 = random.choice(resolutions_temp)

        if random.random() < 0.5:
            positive_img_name = self.image_names[idx]
            pair_path = os.path.join(self.base_dir,res2, positive_img_name)
            label = torch.FloatTensor([0])
        else:
            negative_idx = random.randint(0, len(self.image_names) - 1)
            while negative_idx == idx:
                negative_idx = random.randint(0, len(self.image_names) - 1)
            negative_img_name = self.image_names[negative_idx]
            pair_path = os.path.join(self.base_dir, res2, negative_img_name)
            label = torch.FloatTensor([1])
        
        # Load images
        img1 = Image.open(anchor_path)
        img2 = Image.open(pair_path)

        # Resize images
        if img1.size[0] == 1:
            img1 = transforms.Resize((2,img1.size[1]))(img1)
        
        if img1.size[1] == 1:
            img1 = transforms.Resize((img1.size[0], 2))(img1)

        if img2.size[0] == 1:
            img2 = transforms.Resize((2,img2.size[1]))(img2)
        
        if img2.size[1] == 1:
            img2 = transforms.Resize((img2.size[0], 2))(img2)
        
        # Convert images to RGB
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')


        img1 = self.feature_extractor(images=img1, return_tensors="pt").pixel_values[0]
        img2 = self.feature_extractor(images=img2, return_tensors="pt").pixel_values[0]
        
        
        
        return (img1, img2), label
    
if __name__ == '__main__':
    DATASET_BASE_PATH = './../datasets/flickr8k/'
    train_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
    ])
    
    dataset = SiameseDataset(base_dir=DATASET_BASE_PATH, transform=train_transformations)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(inputs[0].shape, inputs[1].shape, labels)
        break
    