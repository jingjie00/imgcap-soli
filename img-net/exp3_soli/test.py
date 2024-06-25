import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from siamese.SiameseNetwork import Encoder
from torchvision import transforms
from PIL import Image
import numpy as np

# Define image transformations
eval_transformations = transforms.Compose([
    transforms.Resize(256),  # resize shorter side to 256
    transforms.CenterCrop(224),  # center crop to 224x224
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # normalize
])

types = ['flickr8k_images', 'R0.1S1', 'R0.1S50', 'R0.2S1', 'R0.2S50', 'R0.5S1', 'R0.5S50', 'R0.05S50', 'R0.5S1_GF500', 'R1S1_GF500']
base_path = '../datasets/flickr8k/'
image_filenames = ['667626_18933d713e.jpg', '3637013_c675de7705.jpg', '10815824_2997e03d76.jpg', '12830823_87d2654e31.jpg', '17273391_55cfc7d3d4.jpg']
extra_image = ['/flickr8k_images/19212715_20476497a3.jpg',]

path_sets = [[f'{base_path}/{p}/{img}' for p in types] + [f'{base_path}{extra_image}'for p in types] for img in image_filenames]

# Load encoder model
encoder = Encoder(embed_size=50)
encoder.load_state_dict(torch.load('./siamese/saved/resnet50_siamese_best_val_loss.pt', map_location='cpu')['state_dict'])
encoder.eval()  # set model to evaluation mode

# Ensure model is on the correct device
device = torch.device('cpu')  # change to 'cuda' if using GPU
encoder = encoder.to(device)

# Function to compute distance matrix for a set of images
def compute_distance_matrix(image_paths):
    images = [eval_transformations(Image.open(img_path).convert('RGB')).unsqueeze(0) for img_path in image_paths]
    images = [img.to(device) for img in images]
    
    with torch.no_grad():  # disable gradient computation for evaluation
        embeddings = [encoder(img) for img in images]

    num_images = len(images)
    distance_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = nn.functional.pairwise_distance(embeddings[i], embeddings[j]).item()
    
    return distance_matrix

# Compute distance matrices for all sets
distance_matrices = [compute_distance_matrix(paths) for paths in path_sets]

# Compute the average distance matrix
average_distance_matrix = np.mean(distance_matrices, axis=0)

# Function to plot a distance matrix with color
def plot_distance_matrix(matrix, title, labels, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.4g}', ha='center', va='center', color='white' if val < np.max(matrix)/2 else 'black')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

# Create directory to save the images
output_dir = 'distance_matrices'
os.makedirs(output_dir, exist_ok=True)

# Plot and save individual distance matrices
for idx, matrix in enumerate(distance_matrices):
    output_path = os.path.join(output_dir, f'distance_matrix_{idx + 1}.png')
    plot_distance_matrix(matrix, f'Matrix {idx + 1}', types + ['extra'], output_path)

# Plot and save the average distance matrix
output_path = os.path.join(output_dir, 'average_distance_matrix.png')
plot_distance_matrix(average_distance_matrix, 'Average Matrix', types + ['flickr_extra'], output_path)

print("All distance matrices have been saved successfully.")
